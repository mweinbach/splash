#pragma once

#include "MetalBackend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace splash::metal {

inline bool queryPlacementSparseSupport(id<MTLDevice> device) {
    if (@available(macOS 26.4, *)) {
        @try {
            return [device respondsToSelector:@selector(supportsPlacementSparse)] &&
                   device.supportsPlacementSparse;
        } @catch (NSException *exception) {
            // A driver wrapper may expose the selector but fail when forwarding it.
            throw MetalBackendError(
                std::string("Metal supportsPlacementSparse query failed: ") +
                (exception.name.UTF8String ?: "NSException") + ": " +
                (exception.reason.UTF8String ?: "unknown driver error"));
        }
    }
    return false;
}

} // namespace splash::metal
