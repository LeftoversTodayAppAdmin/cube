use crate::wrappers::object::NativeObjectHolder;
use neon::prelude::*;

pub trait NeonSerialize {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>>;
}

impl NeonSerialize for String {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.string(self).upcast::<JsValue>())
    }
}

impl NeonSerialize for i64 {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.number(self.to_owned() as f64).upcast::<JsValue>())
    }
}

impl NeonSerialize for i32 {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.number(self.to_owned() as f64).upcast::<JsValue>())
    }
}

impl NeonSerialize for u64 {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.number(self.to_owned() as f64).upcast::<JsValue>())
    }
}

impl NeonSerialize for u32 {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.number(self.to_owned() as f64).upcast::<JsValue>())
    }
}

impl NeonSerialize for f64 {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.number(self.to_owned()).upcast::<JsValue>())
    }
}

impl NeonSerialize for f32 {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.number(self.to_owned() as f64).upcast::<JsValue>())
    }
}

impl NeonSerialize for bool {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(cx.boolean(self.to_owned()).upcast::<JsValue>())
    }
}

impl<T: NativeObjectHolder> NeonSerialize for T {
    fn to_neon<'a, C: Context<'a>>(&self, cx: &mut C) -> NeonResult<Handle<'a, JsValue>> {
        Ok(self
            .get_native_object()
            .get_object()
            .to_inner(cx)
            .upcast::<JsValue>())
    }
}
