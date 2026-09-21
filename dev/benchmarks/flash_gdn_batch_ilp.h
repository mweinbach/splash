#pragma once
#include "metal/abi/FlashGDN.h"

// gdn.rows is the physical flat input row stride. Actual rows are independent
// per request slot, including masked zero-row slots.
struct GdnBatchIlpParams {
  FlashGDNParams gdn;
  uint32_t actual_rows[4];
};
static_assert(sizeof(GdnBatchIlpParams) == 64);
