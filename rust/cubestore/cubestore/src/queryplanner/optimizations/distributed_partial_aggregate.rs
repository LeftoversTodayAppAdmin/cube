use crate::queryplanner::planning::WorkerExec;
use crate::queryplanner::query_executor::ClusterSendExec;
use crate::queryplanner::tail_limit::TailLimitExec;
use datafusion::error::DataFusionError;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use std::sync::Arc;

/// Transforms from:
///     AggregateFinal
///     `- AggregatePartial
///        `- ClusterSend
/// to:
///     AggregateFinal
///     `- ClusterSend
///        `- AggregatePartial
///
/// The latter gives results in more parallelism and less network.
pub fn push_aggregate_to_workers(
    p_final: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    let p_final_agg: &AggregateExec;
    let p_partial: &Arc<dyn ExecutionPlan>;
    if let Some(a) = p_final.as_any().downcast_ref::<AggregateExec>() {
        if matches!(a.mode(), AggregateMode::Final | AggregateMode::FinalPartitioned) {
            p_final_agg = a;
            p_partial = a.input();
        } else {
            return Ok(p_final);
        }
    } else {
        return Ok(p_final);
    }

    let agg;
    if let Some(a) = p_partial.as_any().downcast_ref::<AggregateExec>() {
        agg = a;
    } else {
        return Ok(p_final);
    }
    if *agg.mode() != AggregateMode::Partial {
        return Ok(p_final);
    }

    let p_final_input: Arc<dyn ExecutionPlan> = if let Some(cs) = agg.input().as_any().downcast_ref::<ClusterSendExec>() {
        let clustersend_input = p_partial.clone()
                    .with_new_children(vec![cs.input_for_optimizations.clone()])?;

        // Note that required_input_ordering is applicable when p_final_agg has a Sorted input mode.

        // Router plan, replace partial aggregate with cluster send.
        Arc::new(
            cs.with_changed_schema(
                clustersend_input,
                p_final_agg.required_input_ordering().into_iter().next().unwrap(),
            ),
        )
    } else if let Some(w) = agg.input().as_any().downcast_ref::<WorkerExec>() {
        let worker_input = p_partial.clone().with_new_children(vec![w.input.clone()])?;

        // Worker plan, execute partial aggregate inside the worker.
        Arc::new(WorkerExec::new(
            worker_input,
            w.max_batch_rows,
            w.limit_and_reverse.clone(),
            p_final_agg.required_input_ordering().into_iter().next().unwrap(),
        ))
    } else {
        return Ok(p_final);
    };

    // We change AggregateMode::FinalPartitioned to AggregateMode::Final, because the ClusterSend
    // node ends up creating an incompatible partitioning for FinalPartitioned.  Some other ideas,
    // like adding a RepartitionExec node, would just be redundant with the behavior of
    // AggregateExec::Final, and also, tricky to set up with the ideal number of partitions in the
    // middle of optimization passes.  Having ClusterSend be able to pass through hash partitions in
    // some form is another option.
    let p_final_input_schema = p_final_input.schema();
    Ok(Arc::new(AggregateExec::try_new(
        AggregateMode::Final,
        p_final_agg.group_expr().clone(),
        p_final_agg.aggr_expr().to_vec(),
        p_final_agg.filter_expr().to_vec(),
        p_final_input,
        p_final_input_schema,
    )?))
}

// TODO upgrade DF: this one was handled by something else but most likely only in sorted scenario
pub fn ensure_partition_merge(
    p: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    if p.as_any().is::<ClusterSendExec>()
        || p.as_any().is::<WorkerExec>()
        || p.as_any().is::<UnionExec>()
    {
        if let Some(ordering) = p.output_ordering() {
            let ordering = ordering.to_vec();
            let merged_children = p
                .children()
                .into_iter()
                .map(|c| -> Arc<dyn ExecutionPlan> {
                    Arc::new(SortPreservingMergeExec::new(ordering.clone(), c.clone()))
                })
                .collect();
            let new_plan = p.with_new_children(merged_children)?;
            Ok(Arc::new(SortPreservingMergeExec::new(ordering, new_plan)))
        } else {
            let merged_children = p
                .children()
                .into_iter()
                .map(|c| -> Arc<dyn ExecutionPlan> {
                    Arc::new(CoalescePartitionsExec::new(c.clone()))
                })
                .collect();
            let new_plan = p.with_new_children(merged_children)?;
            Ok(Arc::new(CoalescePartitionsExec::new(new_plan)))
        }
    } else {
        Ok(p)
    }
}

///Add `GlobalLimitExec` behind worker node if this node has `limit` property set
///Should be executed after all optimizations which can move `Worker` node or change it input
pub fn add_limit_to_workers(
    p: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
    if let Some(w) = p.as_any().downcast_ref::<WorkerExec>() {
        if let Some((limit, reverse)) = w.limit_and_reverse {
            if reverse {
                let limit = Arc::new(TailLimitExec::new(w.input.clone(), limit));
                p.with_new_children(vec![limit])
            } else {
                let limit = Arc::new(GlobalLimitExec::new(w.input.clone(), 0, Some(limit)));
                p.with_new_children(vec![limit])
            }
        } else {
            Ok(p)
        }
    } else {
        Ok(p)
    }
}
