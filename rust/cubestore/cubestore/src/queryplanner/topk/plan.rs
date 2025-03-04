use crate::queryplanner::planning::{ClusterSendNode, CubeExtensionPlanner};
use crate::queryplanner::pretty_printers::pp_plan;
// use crate::queryplanner::topk::execute::{AggregateTopKExec, TopKAggregateFunction};
use crate::queryplanner::topk::{ClusterAggregateTopK, SortColumn, MIN_TOPK_STREAM_ROWS};
use crate::queryplanner::udfs::{
    aggregate_kind_by_name, scalar_udf_by_kind, CubeAggregateUDFKind,
    CubeScalarUDFKind,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::error::DataFusionError;
use datafusion::execution::SessionState;
use datafusion::logical_expr::expr::{AggregateFunction, Alias, ScalarFunction};
use datafusion::physical_plan::expressions::{Column, PhysicalSortExpr};
use datafusion::physical_plan::udf::create_physical_expr;
use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr};

use datafusion::common::{DFSchema, DFSchemaRef};
use datafusion::logical_expr::{Aggregate, Extension, Filter, Limit, LogicalPlan, Projection, SortExpr};
use datafusion::physical_planner::PhysicalPlanner;
use datafusion::prelude::Expr;
use datafusion::sql::TableReference;
use itertools::Itertools;
use std::cmp::max;
use std::sync::Arc;

/// Replaces `Limit(Sort(Aggregate(ClusterSend)))` with [ClusterAggregateTopK] when possible.
pub fn materialize_topk(p: LogicalPlan) -> Result<LogicalPlan, DataFusionError> {
    match &p {
        LogicalPlan::Limit(Limit {
            skip,
            fetch: Some(limit),
            input: sort,
        }) => match sort.as_ref() {
            LogicalPlan::Sort(datafusion::logical_expr::Sort {
                expr: sort_expr,
                input: sort_input,
                fetch: sort_fetch,
            }) => {
                let skip_limit = *skip + *limit;
                let fetch = sort_fetch.unwrap_or(skip_limit).min(skip_limit);
                match materialize_topk_under_limit_sort(fetch, sort_expr, sort_input)? {
                    Some(topk_plan) => return Ok(if *skip == 0 {
                        topk_plan
                    } else {
                        LogicalPlan::Limit(Limit {
                            skip: *skip,
                            fetch: Some(fetch.saturating_sub(*skip)),
                            input: Arc::new(topk_plan),
                        })
                    }),
                    None => {},
                }
            }
            _ => {}
        },
        LogicalPlan::Sort(datafusion::logical_expr::Sort {
            expr: sort_expr,
            input: sort_input,
            fetch: Some(limit),
        }) => {
            match materialize_topk_under_limit_sort(*limit, sort_expr, sort_input)? {
                Some(plan) => return Ok(plan),
                None => {},
            }
        }
        _ => {}
    }

    Ok(p)
}

/// Returns Ok(None) when materialization failed (without error) and the original plan should be returned.
fn materialize_topk_under_limit_sort(fetch: usize, sort_expr: &Vec<SortExpr>, sort_input: &Arc<LogicalPlan>) -> Result<Option<LogicalPlan>, DataFusionError> {
    let projection = extract_projection_and_having(&sort_input);

    let aggregate = projection.as_ref().map(|p| p.input).unwrap_or(sort_input);
    match aggregate.as_ref() {
        LogicalPlan::Aggregate(Aggregate {
            input: cluster_send,
            group_expr,
            aggr_expr,
            schema: aggregate_schema,
            ..
        }) => {
            assert_eq!(
                aggregate_schema.fields().len(),
                group_expr.len() + aggr_expr.len()
            );
            if group_expr.len() == 0
                || aggr_expr.len() == 0
                || !aggr_exprs_allow_topk(aggr_expr)
                || !aggr_schema_allows_topk(aggregate_schema.as_ref(), group_expr.len())
            {
                return Ok(None);
            }
            let sort_columns;
            if let Some(sc) = extract_sort_columns(
                group_expr.len(),
                &sort_expr,
                sort_input.schema(),
                projection.as_ref().map(|c| c.input_columns.as_slice()),
            ) {
                sort_columns = sc;
            } else {
                return Ok(None);
            }
            match cluster_send.as_ref() {
                LogicalPlan::Extension(Extension { node }) => {
                    let cs;
                    if let Some(c) = node.as_any().downcast_ref::<ClusterSendNode>() {
                        cs = c;
                    } else {
                        return Ok(None);
                    }
                    let topk = LogicalPlan::Extension(Extension {
                        node: Arc::new(ClusterAggregateTopK {
                            limit: fetch,
                            input: cs.input.clone(),
                            group_expr: group_expr.clone(),
                            aggregate_expr: aggr_expr.clone(),
                            order_by: sort_columns,
                            having_expr: projection
                                .as_ref()
                                .map_or(None, |p| p.having_expr.clone()),
                            schema: aggregate_schema.clone(),
                            snapshots: cs.snapshots.clone(),
                        }),
                    });
                    if let Some(p) = projection {
                        let in_schema = topk.schema();
                        let out_schema = p.schema;
                        let mut expr = Vec::with_capacity(p.input_columns.len());
                        for out_i in 0..p.input_columns.len() {
                            let in_field = in_schema.field(p.input_columns[out_i]);
                            let out_name: &str = out_schema.field(out_i).name();

                            //let mut e = Expr::Column(f.qualified_column());
                            let mut e =
                                p.post_projection[p.input_columns[out_i]].clone();
                            if out_name != in_field.name() {
                                // TODO upgrade DF: Check if we might need relation -- there is a commented line f.qualified_column() above, too.
                                e = Expr::Alias(Alias { expr: Box::new(e), relation: None, name: out_name.to_owned() });
                            }
                            expr.push(e);
                        }
                        return Ok(Some(LogicalPlan::Projection(Projection::try_new_with_schema(
                            expr,
                            Arc::new(topk),
                            p.schema.clone(),
                        )?)));
                    } else {
                        return Ok(Some(topk));
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }

    Ok(None)
}

fn aggr_exprs_allow_topk(agg_exprs: &[Expr]) -> bool {
    for a in agg_exprs {
        match a {
            // TODO upgrade DF: Deal with planning not creating filter/order_by.
            Expr::AggregateFunction(AggregateFunction { func, args: _, distinct, filter: None, order_by: None, null_treatment: _, .. }) => {
                if *distinct || !fun_allows_topk(func.as_ref()) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    return true;
}

fn aggr_schema_allows_topk(schema: &DFSchema, group_expr_len: usize) -> bool {
    for agg_field in &schema.fields()[group_expr_len..] {
        match agg_field.data_type() {
            DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::Binary
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _) => {} // ok, continue.
            _ => return false,
        }
    }
    return true;
}

fn fun_allows_topk(f: &datafusion::logical_expr::AggregateUDF) -> bool {
    // TODO upgrade DF: Come on, use f.as_any()
    let name = f.name();
    return name == "sum" || name == "min" || name == "max" || name == "merge";

    // Only monotone functions are allowed in principle.
    // Implementation also requires accumulator state and final value to be the same.

    // TODO: lift the restriction and add support for Avg.
    // match f {
    //     AggregateFunction::Sum | AggregateFunction::Min | AggregateFunction::Max => true,
    //     AggregateFunction::Count | AggregateFunction::Avg => false,
    // }
}

// fn extract_aggregate_fun(e: &Expr) -> Option<TopKAggregateFunction> {
//     match e {
//         Expr::AggregateFunction { fun, .. } => match fun {
//             AggregateFunction::Sum => Some(TopKAggregateFunction::Sum),
//             AggregateFunction::Min => Some(TopKAggregateFunction::Min),
//             AggregateFunction::Max => Some(TopKAggregateFunction::Max),
//             _ => None,
//         },
//         Expr::AggregateUDF { fun, .. } => match aggregate_kind_by_name(&fun.name) {
//             Some(CubeAggregateUDFKind::MergeHll) => Some(TopKAggregateFunction::Merge),
//             _ => None,
//         },
//         _ => None,
//     }
// }

#[derive(Debug)]
struct ColumnProjection<'a> {
    input_columns: Vec<usize>,
    input: &'a Arc<LogicalPlan>,
    schema: &'a DFSchemaRef,
    post_projection: Vec<Expr>,
    having_expr: Option<Expr>,
}

fn extract_having(p: &Arc<LogicalPlan>) -> (Option<Expr>, &Arc<LogicalPlan>) {
    // TODO upgrade DF: Filter now has a "having" clause.
    match p.as_ref() {
        LogicalPlan::Filter(Filter { predicate, input, having: _, .. }) => (Some(predicate.clone()), input),
        _ => (None, p),
    }
}

fn extract_projection_and_having(p: &LogicalPlan) -> Option<ColumnProjection> {
    match p {
        LogicalPlan::Projection(Projection {
            expr,
            input,
            schema,
            ..
        }) => {
            let in_schema = input.schema();
            let mut input_columns = Vec::with_capacity(expr.len());
            let mut post_projection = Vec::with_capacity(expr.len());
            for e in expr {
                fn make_column(in_field_qualifier: Option<&TableReference>, in_field: &Field) -> datafusion::common::Column {
                    datafusion::common::Column {
                        relation: in_field_qualifier.map(|tr| tr.clone()),
                        name: in_field.name().clone(),
                    }
                }
                match e {
                    Expr::Alias(Alias { expr: box Expr::Column(c), relation: _, name: _ }) | Expr::Column(c) => {
                        let fi = field_index(in_schema, c.relation.as_ref(), &c.name)?;
                        input_columns.push(fi);
                        let (in_field_qualifier, in_field) = in_schema.qualified_field(fi);
                        post_projection.push(Expr::Column(make_column(in_field_qualifier, in_field)));
                    }
                    Expr::Alias(Alias { expr: box Expr::ScalarFunction(ScalarFunction { func, args }), relation: _, name: _ })
                    | Expr::ScalarFunction(ScalarFunction { func, args }) => match func.name() {
                        // TODO upgrade DF: use as_any() or something
                        "cardinality" => match &args[0] {
                            Expr::Column(c) => {
                                let fi = field_index(in_schema, c.relation.as_ref(), &c.name)?;
                                input_columns.push(fi);
                                let (in_field_qualifier, in_field) = in_schema.qualified_field(fi);
                                post_projection.push(Expr::ScalarFunction(ScalarFunction {
                                    func: scalar_udf_by_kind(CubeScalarUDFKind::HllCardinality),
                                    args: vec![Expr::Column(make_column(in_field_qualifier, in_field))],
                                }));
                            }
                            _ => return None,
                        },
                        _ => return None,
                    },

                    _ => return None,
                }
            }
            let (having_expr, input) = extract_having(input);
            Some(ColumnProjection {
                input_columns,
                input,
                schema,
                post_projection,
                having_expr,
            })
        }
        _ => None,
    }
}

fn extract_sort_columns(
    group_key_len: usize,
    sort_expr: &[SortExpr],
    schema: &DFSchema,
    projection: Option<&[usize]>,
) -> Option<Vec<SortColumn>> {
    let mut sort_columns = Vec::with_capacity(sort_expr.len());
    for e in sort_expr {
        let SortExpr {expr, asc, nulls_first} = e;
        match expr {
            Expr::Column(c) => {
                let mut index = field_index(schema, c.relation.as_ref(), &c.name)?;
                if let Some(p) = projection {
                    index = p[index];
                }
                if index < group_key_len {
                    return None;
                }
                sort_columns.push(SortColumn {
                    agg_index: index - group_key_len,
                    asc: *asc,
                    nulls_first: *nulls_first,
                })
            }
            _ => return None,
        }
    }
    Some(sort_columns)
}

fn field_index(schema: &DFSchema, qualifier: Option<&TableReference>, name: &str) -> Option<usize> {
    schema.index_of_column_by_name(qualifier, name)

    // TODO upgrade DF: Reconsider.
    // schema
    //     .iter()
    //     .position(|f| f.qualifier().map(|s| s.as_str()) == qualifier && f.name() == name)
}

// pub fn plan_topk(
//     planner: &dyn PhysicalPlanner,
//     ext_planner: &CubeExtensionPlanner,
//     node: &ClusterAggregateTopK,
//     input: Arc<dyn ExecutionPlan>,
//     ctx: &ExecutionContextState,
// ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
//     // Partial aggregate on workers. Mimics corresponding planning code from DataFusion.
//     let physical_input_schema = input.schema();
//     let logical_input_schema = node.input.schema();
//     let group_expr = node
//         .group_expr
//         .iter()
//         .map(|e| {
//             Ok((
//                 planner.create_physical_expr(
//                     e,
//                     &logical_input_schema,
//                     &physical_input_schema,
//                     ctx,
//                 )?,
//                 physical_name(e, &logical_input_schema)?,
//             ))
//         })
//         .collect::<Result<Vec<_>, DataFusionError>>()?;
//     let group_expr_len = group_expr.len();
//     let initial_aggregate_expr = node
//         .aggregate_expr
//         .iter()
//         .map(|e| {
//             planner.create_aggregate_expr(e, &logical_input_schema, &physical_input_schema, ctx)
//         })
//         .collect::<Result<Vec<_>, DataFusionError>>()?;
//     let (strategy, order) = compute_aggregation_strategy(input.as_ref(), &group_expr);
//     let aggregate = Arc::new(HashAggregateExec::try_new(
//         strategy,
//         order,
//         AggregateMode::Full,
//         group_expr,
//         initial_aggregate_expr.clone(),
//         input,
//         physical_input_schema,
//     )?);
//
//     let aggregate_schema = aggregate.as_ref().schema();
//
//     let agg_fun = node
//         .aggregate_expr
//         .iter()
//         .map(|e| extract_aggregate_fun(e).unwrap())
//         .collect_vec();
//     //
//     // Sort on workers.
//     let sort_expr = node
//         .order_by
//         .iter()
//         .map(|c| {
//             let i = group_expr_len + c.agg_index;
//             PhysicalSortExpr {
//                 expr: make_sort_expr(
//                     &aggregate_schema,
//                     &agg_fun[c.agg_index],
//                     Arc::new(Column::new(aggregate_schema.field(i).name(), i)),
//                 ),
//                 options: SortOptions {
//                     descending: !c.asc,
//                     nulls_first: c.nulls_first,
//                 },
//             }
//         })
//         .collect_vec();
//     let sort = Arc::new(SortExec::try_new(sort_expr, aggregate)?);
//     let sort_schema = sort.schema();
//
//     // Send results to router.
//     let schema = sort_schema.clone();
//     let cluster = ext_planner.plan_cluster_send(
//         sort,
//         &node.snapshots,
//         schema.clone(),
//         /*use_streaming*/ true,
//         /*max_batch_rows*/ max(2 * node.limit, MIN_TOPK_STREAM_ROWS),
//         None,
//     )?;
//
//     let having = if let Some(predicate) = &node.having_expr {
//         Some(planner.create_physical_expr(predicate, &node.schema, &schema, ctx)?)
//     } else {
//         None
//     };
//
//     Ok(Arc::new(AggregateTopKExec::new(
//         node.limit,
//         group_expr_len,
//         initial_aggregate_expr,
//         &agg_fun,
//         node.order_by.clone(),
//         having,
//         cluster,
//         schema,
//     )))
// }
//
// fn make_sort_expr(
//     schema: &Arc<Schema>,
//     fun: &TopKAggregateFunction,
//     col: Arc<dyn PhysicalExpr>,
// ) -> Arc<dyn PhysicalExpr> {
//     match fun {
//         TopKAggregateFunction::Merge => create_physical_expr(
//             &scalar_udf_by_kind(CubeScalarUDFKind::HllCardinality).descriptor(),
//             &[col],
//             schema,
//         )
//         .unwrap(),
//         _ => col,
//     }
// }

// TODO upgrade DF: Remove (or dedup with pp_plan code)
fn pp_topk_line(topk: &ClusterAggregateTopK) -> String {
    let mut output = String::new();
    output += &format!("ClusterAggregateTopK, limit: {}", topk.limit);
    if true /* self.opts.show_aggregations */ {
        output += &format!(", aggs: {:?}", topk.aggregate_expr)
    }
    if true /* self.opts.show_sort_by */ {
        output += &format!(
            ", sortBy: {}",
            crate::queryplanner::pretty_printers::pp_sort_columns(topk.group_expr.len(), &topk.order_by)
        );
    }
    if true /* self.opts.show_filters */ {
        if let Some(having) = &topk.having_expr {
            output += &format!(", having: {:?}", having)
        }
    }
    output
}


// TODO upgrade DF: Remove
/// Turns a topk logical node into a plan -- but simply creates an inefficient plan.  A work in progress.
pub fn plan_topk(
    planner: &dyn PhysicalPlanner,
    ext_planner: &CubeExtensionPlanner,
    node: &ClusterAggregateTopK,
    input: Arc<dyn ExecutionPlan>,
    ctx: &SessionState,
) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {

    println!("Planning topk node: {}\n{}", pp_topk_line(node), pp_plan(node.input.as_ref()));

    // TODO upgrade DF: Implement.
    Err(DataFusionError::NotImplemented("plan_topk not implemented".to_owned()))
}