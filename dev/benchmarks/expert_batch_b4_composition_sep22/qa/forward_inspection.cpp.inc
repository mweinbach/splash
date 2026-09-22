// Private clone only. Original forward/verify/state methods remain unchanged.
namespace splash::flash {
bool FlashDeepPrefixOracle::rawOwns(const FlashForward &target,const FlashRequestState &request){
  return target.impl_&&request.impl_&&request.impl_->owner==target.impl_->owner&&request.impl_->capacity==target.impl_->capacity;
}
}
