; ModuleID = '/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_odd_rowpair_sep22/kernel/_cpu_build_v3/native-qmv.air'
source_filename = "runtime/metal/kernels/shared/flash_affine_qmv_f32.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

%"struct.metal::_atomic" = type { i32 }
%struct.FlashAffineParams = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64 }

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q4_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %77, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %77, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %77, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %77, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %77, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 4
  br i1 %34, label %35, label %77

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 64
  %39 = and i32 %29, 63
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %77

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %77

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %77, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %77

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = lshr i32 %29, 1
  %60 = zext i32 %59 to i64
  %61 = icmp ult i64 %58, %60
  br i1 %61, label %77, label %62

62:                                               ; preds = %56
  %63 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %64 = load i64, i64 addrspace(2)* %63, align 8, !tbaa !53
  %65 = lshr i32 %29, 5
  %66 = and i32 %65, 134217726
  %67 = zext i32 %66 to i64
  %68 = icmp uge i64 %64, %67
  %69 = and i64 %64, 1
  %70 = icmp eq i64 %69, 0
  %71 = and i1 %68, %70
  br i1 %71, label %72, label %77

72:                                               ; preds = %62
  %73 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %74 = load i64, i64 addrspace(2)* %73, align 8, !tbaa !54
  %75 = and i64 %74, 1
  %76 = icmp eq i64 %75, 0
  br i1 %76, label %82, label %77

77:                                               ; preds = %72, %62, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %78 = icmp eq i32 %10, 0
  br i1 %78, label %79, label %142

79:                                               ; preds = %77
  %80 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %81 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %80, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %142

82:                                               ; preds = %72
  %83 = extractelement <3 x i32> %8, i64 0
  %84 = shl i32 %83, 3
  %85 = shl i32 %9, 2
  %86 = add i32 %84, %85
  %87 = icmp ult i32 %86, %25
  br i1 %87, label %88, label %142

88:                                               ; preds = %82
  %89 = extractelement <3 x i32> %8, i64 1
  %90 = icmp ult i32 %89, %13
  br i1 %90, label %91, label %142

91:                                               ; preds = %88
  %92 = extractelement <3 x i32> %8, i64 2
  %93 = icmp ult i32 %92, %17
  br i1 %93, label %94, label %142

94:                                               ; preds = %91
  %95 = zext i32 %89 to i64
  %96 = zext i32 %17 to i64
  %97 = mul nuw i64 %96, %95
  %98 = zext i32 %92 to i64
  %99 = add nuw i64 %97, %98
  br i1 %49, label %103, label %100

100:                                              ; preds = %94
  %101 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %99
  %102 = load i64, i64 addrspace(1)* %101, align 8, !tbaa !55
  br label %103

103:                                              ; preds = %100, %94
  %104 = phi i64 [ %102, %100 ], [ 0, %94 ]
  %105 = icmp sgt i64 %104, -1
  %106 = zext i32 %21 to i64
  %107 = icmp ult i64 %104, %106
  %108 = select i1 %105, i1 %107, i1 false
  br i1 %108, label %129, label %109

109:                                              ; preds = %103
  %110 = icmp eq i32 %10, 0
  br i1 %110, label %111, label %142

111:                                              ; preds = %109
  %112 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %113 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %112, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %114 = zext i32 %25 to i64
  %115 = mul i64 %99, %114
  %116 = zext i32 %86 to i64
  %117 = add i64 %115, %116
  br label %118

118:                                              ; preds = %126, %111
  %119 = phi i32 [ 0, %111 ], [ %127, %126 ]
  %120 = add nuw nsw i32 %119, %86
  %121 = icmp ult i32 %120, %25
  br i1 %121, label %122, label %126

122:                                              ; preds = %118
  %123 = zext i32 %119 to i64
  %124 = add i64 %117, %123
  %125 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %124
  store bfloat 0xR7FC0, bfloat addrspace(1)* %125, align 2, !tbaa !56
  br label %126

126:                                              ; preds = %122, %118
  %127 = add nuw nsw i32 %119, 1
  %128 = icmp eq i32 %127, 4
  br i1 %128, label %142, label %118, !llvm.loop !58

129:                                              ; preds = %103
  %130 = icmp eq i32 %47, 0
  %131 = select i1 %130, i64 %95, i64 %99
  %132 = zext i32 %29 to i64
  %133 = mul i64 %131, %132
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %133
  %135 = icmp ugt i32 %25, 7
  %136 = and i32 %29, 511
  %137 = icmp eq i32 %136, 0
  %138 = select i1 %135, i1 %137, i1 false
  %139 = trunc i64 %104 to i32
  br i1 %138, label %140, label %141

140:                                              ; preds = %129
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %86, i32 noundef %139, i64 noundef %99, i32 noundef %10) #10
  br label %142

141:                                              ; preds = %129
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %86, i32 noundef %139, i64 noundef %99, i32 noundef %10) #10
  br label %142

142:                                              ; preds = %141, %140, %126, %109, %91, %88, %82, %79, %77
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q4_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt4ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt4ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %77, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %77, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %77, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %77, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %77, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 4
  br i1 %34, label %35, label %77

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 128
  %39 = and i32 %29, 127
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %77

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %77

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %77, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %77

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = lshr i32 %29, 1
  %60 = zext i32 %59 to i64
  %61 = icmp ult i64 %58, %60
  br i1 %61, label %77, label %62

62:                                               ; preds = %56
  %63 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %64 = load i64, i64 addrspace(2)* %63, align 8, !tbaa !53
  %65 = lshr i32 %29, 6
  %66 = and i32 %65, 67108862
  %67 = zext i32 %66 to i64
  %68 = icmp uge i64 %64, %67
  %69 = and i64 %64, 1
  %70 = icmp eq i64 %69, 0
  %71 = and i1 %68, %70
  br i1 %71, label %72, label %77

72:                                               ; preds = %62
  %73 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %74 = load i64, i64 addrspace(2)* %73, align 8, !tbaa !54
  %75 = and i64 %74, 1
  %76 = icmp eq i64 %75, 0
  br i1 %76, label %82, label %77

77:                                               ; preds = %72, %62, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %78 = icmp eq i32 %10, 0
  br i1 %78, label %79, label %142

79:                                               ; preds = %77
  %80 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %81 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %80, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %142

82:                                               ; preds = %72
  %83 = extractelement <3 x i32> %8, i64 0
  %84 = shl i32 %83, 3
  %85 = shl i32 %9, 2
  %86 = add i32 %84, %85
  %87 = icmp ult i32 %86, %25
  br i1 %87, label %88, label %142

88:                                               ; preds = %82
  %89 = extractelement <3 x i32> %8, i64 1
  %90 = icmp ult i32 %89, %13
  br i1 %90, label %91, label %142

91:                                               ; preds = %88
  %92 = extractelement <3 x i32> %8, i64 2
  %93 = icmp ult i32 %92, %17
  br i1 %93, label %94, label %142

94:                                               ; preds = %91
  %95 = zext i32 %89 to i64
  %96 = zext i32 %17 to i64
  %97 = mul nuw i64 %96, %95
  %98 = zext i32 %92 to i64
  %99 = add nuw i64 %97, %98
  br i1 %49, label %103, label %100

100:                                              ; preds = %94
  %101 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %99
  %102 = load i64, i64 addrspace(1)* %101, align 8, !tbaa !55
  br label %103

103:                                              ; preds = %100, %94
  %104 = phi i64 [ %102, %100 ], [ 0, %94 ]
  %105 = icmp sgt i64 %104, -1
  %106 = zext i32 %21 to i64
  %107 = icmp ult i64 %104, %106
  %108 = select i1 %105, i1 %107, i1 false
  br i1 %108, label %129, label %109

109:                                              ; preds = %103
  %110 = icmp eq i32 %10, 0
  br i1 %110, label %111, label %142

111:                                              ; preds = %109
  %112 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %113 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %112, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %114 = zext i32 %25 to i64
  %115 = mul i64 %99, %114
  %116 = zext i32 %86 to i64
  %117 = add i64 %115, %116
  br label %118

118:                                              ; preds = %126, %111
  %119 = phi i32 [ 0, %111 ], [ %127, %126 ]
  %120 = add nuw nsw i32 %119, %86
  %121 = icmp ult i32 %120, %25
  br i1 %121, label %122, label %126

122:                                              ; preds = %118
  %123 = zext i32 %119 to i64
  %124 = add i64 %117, %123
  %125 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %124
  store bfloat 0xR7FC0, bfloat addrspace(1)* %125, align 2, !tbaa !56
  br label %126

126:                                              ; preds = %122, %118
  %127 = add nuw nsw i32 %119, 1
  %128 = icmp eq i32 %127, 4
  br i1 %128, label %142, label %118, !llvm.loop !60

129:                                              ; preds = %103
  %130 = icmp eq i32 %47, 0
  %131 = select i1 %130, i64 %95, i64 %99
  %132 = zext i32 %29 to i64
  %133 = mul i64 %131, %132
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %133
  %135 = icmp ugt i32 %25, 7
  %136 = and i32 %29, 511
  %137 = icmp eq i32 %136, 0
  %138 = select i1 %135, i1 %137, i1 false
  %139 = trunc i64 %104 to i32
  br i1 %138, label %140, label %141

140:                                              ; preds = %129
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %86, i32 noundef %139, i64 noundef %99, i32 noundef %10) #10
  br label %142

141:                                              ; preds = %129
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %86, i32 noundef %139, i64 noundef %99, i32 noundef %10) #10
  br label %142

142:                                              ; preds = %141, %140, %126, %109, %91, %88, %82, %79, %77
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q5_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %78, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %78, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %78, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %78, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %78, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 5
  br i1 %34, label %35, label %78

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 64
  %39 = and i32 %29, 63
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %78

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %78

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %78, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %78

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = zext i32 %29 to i64
  %60 = mul nuw nsw i64 %59, 5
  %61 = lshr i64 %60, 3
  %62 = icmp ult i64 %58, %61
  br i1 %62, label %78, label %63

63:                                               ; preds = %56
  %64 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %65 = load i64, i64 addrspace(2)* %64, align 8, !tbaa !53
  %66 = lshr i32 %29, 5
  %67 = and i32 %66, 134217726
  %68 = zext i32 %67 to i64
  %69 = icmp uge i64 %65, %68
  %70 = and i64 %65, 1
  %71 = icmp eq i64 %70, 0
  %72 = and i1 %69, %71
  br i1 %72, label %73, label %78

73:                                               ; preds = %63
  %74 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %75 = load i64, i64 addrspace(2)* %74, align 8, !tbaa !54
  %76 = and i64 %75, 1
  %77 = icmp eq i64 %76, 0
  br i1 %77, label %83, label %78

78:                                               ; preds = %73, %63, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %79 = icmp eq i32 %10, 0
  br i1 %79, label %80, label %142

80:                                               ; preds = %78
  %81 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %82 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %81, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %142

83:                                               ; preds = %73
  %84 = extractelement <3 x i32> %8, i64 0
  %85 = shl i32 %84, 3
  %86 = shl i32 %9, 2
  %87 = add i32 %85, %86
  %88 = icmp ult i32 %87, %25
  br i1 %88, label %89, label %142

89:                                               ; preds = %83
  %90 = extractelement <3 x i32> %8, i64 1
  %91 = icmp ult i32 %90, %13
  br i1 %91, label %92, label %142

92:                                               ; preds = %89
  %93 = extractelement <3 x i32> %8, i64 2
  %94 = icmp ult i32 %93, %17
  br i1 %94, label %95, label %142

95:                                               ; preds = %92
  %96 = zext i32 %90 to i64
  %97 = zext i32 %17 to i64
  %98 = mul nuw i64 %97, %96
  %99 = zext i32 %93 to i64
  %100 = add nuw i64 %98, %99
  br i1 %49, label %104, label %101

101:                                              ; preds = %95
  %102 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %100
  %103 = load i64, i64 addrspace(1)* %102, align 8, !tbaa !55
  br label %104

104:                                              ; preds = %101, %95
  %105 = phi i64 [ %103, %101 ], [ 0, %95 ]
  %106 = icmp sgt i64 %105, -1
  %107 = zext i32 %21 to i64
  %108 = icmp ult i64 %105, %107
  %109 = select i1 %106, i1 %108, i1 false
  br i1 %109, label %130, label %110

110:                                              ; preds = %104
  %111 = icmp eq i32 %10, 0
  br i1 %111, label %112, label %142

112:                                              ; preds = %110
  %113 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %114 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %113, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %115 = zext i32 %25 to i64
  %116 = mul i64 %100, %115
  %117 = zext i32 %87 to i64
  %118 = add i64 %116, %117
  br label %119

119:                                              ; preds = %127, %112
  %120 = phi i32 [ 0, %112 ], [ %128, %127 ]
  %121 = add nuw nsw i32 %120, %87
  %122 = icmp ult i32 %121, %25
  br i1 %122, label %123, label %127

123:                                              ; preds = %119
  %124 = zext i32 %120 to i64
  %125 = add i64 %118, %124
  %126 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %125
  store bfloat 0xR7FC0, bfloat addrspace(1)* %126, align 2, !tbaa !56
  br label %127

127:                                              ; preds = %123, %119
  %128 = add nuw nsw i32 %120, 1
  %129 = icmp eq i32 %128, 4
  br i1 %129, label %142, label %119, !llvm.loop !61

130:                                              ; preds = %104
  %131 = icmp eq i32 %47, 0
  %132 = select i1 %131, i64 %96, i64 %100
  %133 = mul i64 %132, %59
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %133
  %135 = icmp ugt i32 %25, 7
  %136 = and i32 %29, 511
  %137 = icmp eq i32 %136, 0
  %138 = select i1 %135, i1 %137, i1 false
  %139 = trunc i64 %105 to i32
  br i1 %138, label %140, label %141

140:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

141:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

142:                                              ; preds = %141, %140, %127, %110, %92, %89, %83, %80, %78
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q5_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %78, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %78, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %78, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %78, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %78, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 5
  br i1 %34, label %35, label %78

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 128
  %39 = and i32 %29, 127
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %78

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %78

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %78, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %78

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = zext i32 %29 to i64
  %60 = mul nuw nsw i64 %59, 5
  %61 = lshr i64 %60, 3
  %62 = icmp ult i64 %58, %61
  br i1 %62, label %78, label %63

63:                                               ; preds = %56
  %64 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %65 = load i64, i64 addrspace(2)* %64, align 8, !tbaa !53
  %66 = lshr i32 %29, 6
  %67 = and i32 %66, 67108862
  %68 = zext i32 %67 to i64
  %69 = icmp uge i64 %65, %68
  %70 = and i64 %65, 1
  %71 = icmp eq i64 %70, 0
  %72 = and i1 %69, %71
  br i1 %72, label %73, label %78

73:                                               ; preds = %63
  %74 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %75 = load i64, i64 addrspace(2)* %74, align 8, !tbaa !54
  %76 = and i64 %75, 1
  %77 = icmp eq i64 %76, 0
  br i1 %77, label %83, label %78

78:                                               ; preds = %73, %63, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %79 = icmp eq i32 %10, 0
  br i1 %79, label %80, label %142

80:                                               ; preds = %78
  %81 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %82 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %81, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %142

83:                                               ; preds = %73
  %84 = extractelement <3 x i32> %8, i64 0
  %85 = shl i32 %84, 3
  %86 = shl i32 %9, 2
  %87 = add i32 %85, %86
  %88 = icmp ult i32 %87, %25
  br i1 %88, label %89, label %142

89:                                               ; preds = %83
  %90 = extractelement <3 x i32> %8, i64 1
  %91 = icmp ult i32 %90, %13
  br i1 %91, label %92, label %142

92:                                               ; preds = %89
  %93 = extractelement <3 x i32> %8, i64 2
  %94 = icmp ult i32 %93, %17
  br i1 %94, label %95, label %142

95:                                               ; preds = %92
  %96 = zext i32 %90 to i64
  %97 = zext i32 %17 to i64
  %98 = mul nuw i64 %97, %96
  %99 = zext i32 %93 to i64
  %100 = add nuw i64 %98, %99
  br i1 %49, label %104, label %101

101:                                              ; preds = %95
  %102 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %100
  %103 = load i64, i64 addrspace(1)* %102, align 8, !tbaa !55
  br label %104

104:                                              ; preds = %101, %95
  %105 = phi i64 [ %103, %101 ], [ 0, %95 ]
  %106 = icmp sgt i64 %105, -1
  %107 = zext i32 %21 to i64
  %108 = icmp ult i64 %105, %107
  %109 = select i1 %106, i1 %108, i1 false
  br i1 %109, label %130, label %110

110:                                              ; preds = %104
  %111 = icmp eq i32 %10, 0
  br i1 %111, label %112, label %142

112:                                              ; preds = %110
  %113 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %114 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %113, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %115 = zext i32 %25 to i64
  %116 = mul i64 %100, %115
  %117 = zext i32 %87 to i64
  %118 = add i64 %116, %117
  br label %119

119:                                              ; preds = %127, %112
  %120 = phi i32 [ 0, %112 ], [ %128, %127 ]
  %121 = add nuw nsw i32 %120, %87
  %122 = icmp ult i32 %121, %25
  br i1 %122, label %123, label %127

123:                                              ; preds = %119
  %124 = zext i32 %120 to i64
  %125 = add i64 %118, %124
  %126 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %125
  store bfloat 0xR7FC0, bfloat addrspace(1)* %126, align 2, !tbaa !56
  br label %127

127:                                              ; preds = %123, %119
  %128 = add nuw nsw i32 %120, 1
  %129 = icmp eq i32 %128, 4
  br i1 %129, label %142, label %119, !llvm.loop !62

130:                                              ; preds = %104
  %131 = icmp eq i32 %47, 0
  %132 = select i1 %131, i64 %96, i64 %100
  %133 = mul i64 %132, %59
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %133
  %135 = icmp ugt i32 %25, 7
  %136 = and i32 %29, 511
  %137 = icmp eq i32 %136, 0
  %138 = select i1 %135, i1 %137, i1 false
  %139 = trunc i64 %105 to i32
  br i1 %138, label %140, label %141

140:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

141:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

142:                                              ; preds = %141, %140, %127, %110, %92, %89, %83, %80, %78
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q6_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %78, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %78, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %78, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %78, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %78, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 6
  br i1 %34, label %35, label %78

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 64
  %39 = and i32 %29, 63
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %78

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %78

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %78, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %78

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = zext i32 %29 to i64
  %60 = mul nuw nsw i64 %59, 6
  %61 = lshr i64 %60, 3
  %62 = icmp ult i64 %58, %61
  br i1 %62, label %78, label %63

63:                                               ; preds = %56
  %64 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %65 = load i64, i64 addrspace(2)* %64, align 8, !tbaa !53
  %66 = lshr i32 %29, 5
  %67 = and i32 %66, 134217726
  %68 = zext i32 %67 to i64
  %69 = icmp uge i64 %65, %68
  %70 = and i64 %65, 1
  %71 = icmp eq i64 %70, 0
  %72 = and i1 %69, %71
  br i1 %72, label %73, label %78

73:                                               ; preds = %63
  %74 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %75 = load i64, i64 addrspace(2)* %74, align 8, !tbaa !54
  %76 = and i64 %75, 1
  %77 = icmp eq i64 %76, 0
  br i1 %77, label %83, label %78

78:                                               ; preds = %73, %63, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %79 = icmp eq i32 %10, 0
  br i1 %79, label %80, label %142

80:                                               ; preds = %78
  %81 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %82 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %81, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %142

83:                                               ; preds = %73
  %84 = extractelement <3 x i32> %8, i64 0
  %85 = shl i32 %84, 3
  %86 = shl i32 %9, 2
  %87 = add i32 %85, %86
  %88 = icmp ult i32 %87, %25
  br i1 %88, label %89, label %142

89:                                               ; preds = %83
  %90 = extractelement <3 x i32> %8, i64 1
  %91 = icmp ult i32 %90, %13
  br i1 %91, label %92, label %142

92:                                               ; preds = %89
  %93 = extractelement <3 x i32> %8, i64 2
  %94 = icmp ult i32 %93, %17
  br i1 %94, label %95, label %142

95:                                               ; preds = %92
  %96 = zext i32 %90 to i64
  %97 = zext i32 %17 to i64
  %98 = mul nuw i64 %97, %96
  %99 = zext i32 %93 to i64
  %100 = add nuw i64 %98, %99
  br i1 %49, label %104, label %101

101:                                              ; preds = %95
  %102 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %100
  %103 = load i64, i64 addrspace(1)* %102, align 8, !tbaa !55
  br label %104

104:                                              ; preds = %101, %95
  %105 = phi i64 [ %103, %101 ], [ 0, %95 ]
  %106 = icmp sgt i64 %105, -1
  %107 = zext i32 %21 to i64
  %108 = icmp ult i64 %105, %107
  %109 = select i1 %106, i1 %108, i1 false
  br i1 %109, label %130, label %110

110:                                              ; preds = %104
  %111 = icmp eq i32 %10, 0
  br i1 %111, label %112, label %142

112:                                              ; preds = %110
  %113 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %114 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %113, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %115 = zext i32 %25 to i64
  %116 = mul i64 %100, %115
  %117 = zext i32 %87 to i64
  %118 = add i64 %116, %117
  br label %119

119:                                              ; preds = %127, %112
  %120 = phi i32 [ 0, %112 ], [ %128, %127 ]
  %121 = add nuw nsw i32 %120, %87
  %122 = icmp ult i32 %121, %25
  br i1 %122, label %123, label %127

123:                                              ; preds = %119
  %124 = zext i32 %120 to i64
  %125 = add i64 %118, %124
  %126 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %125
  store bfloat 0xR7FC0, bfloat addrspace(1)* %126, align 2, !tbaa !56
  br label %127

127:                                              ; preds = %123, %119
  %128 = add nuw nsw i32 %120, 1
  %129 = icmp eq i32 %128, 4
  br i1 %129, label %142, label %119, !llvm.loop !63

130:                                              ; preds = %104
  %131 = icmp eq i32 %47, 0
  %132 = select i1 %131, i64 %96, i64 %100
  %133 = mul i64 %132, %59
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %133
  %135 = icmp ugt i32 %25, 7
  %136 = and i32 %29, 255
  %137 = icmp eq i32 %136, 0
  %138 = select i1 %135, i1 %137, i1 false
  %139 = trunc i64 %105 to i32
  br i1 %138, label %140, label %141

140:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

141:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

142:                                              ; preds = %141, %140, %127, %110, %92, %89, %83, %80, %78
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q6_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt6ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt6ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %78, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %78, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %78, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %78, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %78, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 6
  br i1 %34, label %35, label %78

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 128
  %39 = and i32 %29, 127
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %78

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %78

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %78, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %78

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = zext i32 %29 to i64
  %60 = mul nuw nsw i64 %59, 6
  %61 = lshr i64 %60, 3
  %62 = icmp ult i64 %58, %61
  br i1 %62, label %78, label %63

63:                                               ; preds = %56
  %64 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %65 = load i64, i64 addrspace(2)* %64, align 8, !tbaa !53
  %66 = lshr i32 %29, 6
  %67 = and i32 %66, 67108862
  %68 = zext i32 %67 to i64
  %69 = icmp uge i64 %65, %68
  %70 = and i64 %65, 1
  %71 = icmp eq i64 %70, 0
  %72 = and i1 %69, %71
  br i1 %72, label %73, label %78

73:                                               ; preds = %63
  %74 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %75 = load i64, i64 addrspace(2)* %74, align 8, !tbaa !54
  %76 = and i64 %75, 1
  %77 = icmp eq i64 %76, 0
  br i1 %77, label %83, label %78

78:                                               ; preds = %73, %63, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %79 = icmp eq i32 %10, 0
  br i1 %79, label %80, label %142

80:                                               ; preds = %78
  %81 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %82 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %81, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %142

83:                                               ; preds = %73
  %84 = extractelement <3 x i32> %8, i64 0
  %85 = shl i32 %84, 3
  %86 = shl i32 %9, 2
  %87 = add i32 %85, %86
  %88 = icmp ult i32 %87, %25
  br i1 %88, label %89, label %142

89:                                               ; preds = %83
  %90 = extractelement <3 x i32> %8, i64 1
  %91 = icmp ult i32 %90, %13
  br i1 %91, label %92, label %142

92:                                               ; preds = %89
  %93 = extractelement <3 x i32> %8, i64 2
  %94 = icmp ult i32 %93, %17
  br i1 %94, label %95, label %142

95:                                               ; preds = %92
  %96 = zext i32 %90 to i64
  %97 = zext i32 %17 to i64
  %98 = mul nuw i64 %97, %96
  %99 = zext i32 %93 to i64
  %100 = add nuw i64 %98, %99
  br i1 %49, label %104, label %101

101:                                              ; preds = %95
  %102 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %100
  %103 = load i64, i64 addrspace(1)* %102, align 8, !tbaa !55
  br label %104

104:                                              ; preds = %101, %95
  %105 = phi i64 [ %103, %101 ], [ 0, %95 ]
  %106 = icmp sgt i64 %105, -1
  %107 = zext i32 %21 to i64
  %108 = icmp ult i64 %105, %107
  %109 = select i1 %106, i1 %108, i1 false
  br i1 %109, label %130, label %110

110:                                              ; preds = %104
  %111 = icmp eq i32 %10, 0
  br i1 %111, label %112, label %142

112:                                              ; preds = %110
  %113 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %114 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %113, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %115 = zext i32 %25 to i64
  %116 = mul i64 %100, %115
  %117 = zext i32 %87 to i64
  %118 = add i64 %116, %117
  br label %119

119:                                              ; preds = %127, %112
  %120 = phi i32 [ 0, %112 ], [ %128, %127 ]
  %121 = add nuw nsw i32 %120, %87
  %122 = icmp ult i32 %121, %25
  br i1 %122, label %123, label %127

123:                                              ; preds = %119
  %124 = zext i32 %120 to i64
  %125 = add i64 %118, %124
  %126 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %125
  store bfloat 0xR7FC0, bfloat addrspace(1)* %126, align 2, !tbaa !56
  br label %127

127:                                              ; preds = %123, %119
  %128 = add nuw nsw i32 %120, 1
  %129 = icmp eq i32 %128, 4
  br i1 %129, label %142, label %119, !llvm.loop !64

130:                                              ; preds = %104
  %131 = icmp eq i32 %47, 0
  %132 = select i1 %131, i64 %96, i64 %100
  %133 = mul i64 %132, %59
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %133
  %135 = icmp ugt i32 %25, 7
  %136 = and i32 %29, 255
  %137 = icmp eq i32 %136, 0
  %138 = select i1 %135, i1 %137, i1 false
  %139 = trunc i64 %105 to i32
  br i1 %138, label %140, label %141

140:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

141:                                              ; preds = %130
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %134, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %87, i32 noundef %139, i64 noundef %100, i32 noundef %10) #10
  br label %142

142:                                              ; preds = %141, %140, %127, %110, %92, %89, %83, %80, %78
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q8_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt8ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt8ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %76, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %76, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %76, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %76, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %76, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 8
  br i1 %34, label %35, label %76

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 64
  %39 = and i32 %29, 63
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %76

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %76

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %76, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %76

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = zext i32 %29 to i64
  %60 = icmp ult i64 %58, %59
  br i1 %60, label %76, label %61

61:                                               ; preds = %56
  %62 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %63 = load i64, i64 addrspace(2)* %62, align 8, !tbaa !53
  %64 = lshr i32 %29, 5
  %65 = and i32 %64, 134217726
  %66 = zext i32 %65 to i64
  %67 = icmp uge i64 %63, %66
  %68 = and i64 %63, 1
  %69 = icmp eq i64 %68, 0
  %70 = and i1 %67, %69
  br i1 %70, label %71, label %76

71:                                               ; preds = %61
  %72 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %73 = load i64, i64 addrspace(2)* %72, align 8, !tbaa !54
  %74 = and i64 %73, 1
  %75 = icmp eq i64 %74, 0
  br i1 %75, label %81, label %76

76:                                               ; preds = %71, %61, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %77 = icmp eq i32 %10, 0
  br i1 %77, label %78, label %140

78:                                               ; preds = %76
  %79 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %80 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %79, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %140

81:                                               ; preds = %71
  %82 = extractelement <3 x i32> %8, i64 0
  %83 = shl i32 %82, 3
  %84 = shl i32 %9, 2
  %85 = add i32 %83, %84
  %86 = icmp ult i32 %85, %25
  br i1 %86, label %87, label %140

87:                                               ; preds = %81
  %88 = extractelement <3 x i32> %8, i64 1
  %89 = icmp ult i32 %88, %13
  br i1 %89, label %90, label %140

90:                                               ; preds = %87
  %91 = extractelement <3 x i32> %8, i64 2
  %92 = icmp ult i32 %91, %17
  br i1 %92, label %93, label %140

93:                                               ; preds = %90
  %94 = zext i32 %88 to i64
  %95 = zext i32 %17 to i64
  %96 = mul nuw i64 %95, %94
  %97 = zext i32 %91 to i64
  %98 = add nuw i64 %96, %97
  br i1 %49, label %102, label %99

99:                                               ; preds = %93
  %100 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %98
  %101 = load i64, i64 addrspace(1)* %100, align 8, !tbaa !55
  br label %102

102:                                              ; preds = %99, %93
  %103 = phi i64 [ %101, %99 ], [ 0, %93 ]
  %104 = icmp sgt i64 %103, -1
  %105 = zext i32 %21 to i64
  %106 = icmp ult i64 %103, %105
  %107 = select i1 %104, i1 %106, i1 false
  br i1 %107, label %128, label %108

108:                                              ; preds = %102
  %109 = icmp eq i32 %10, 0
  br i1 %109, label %110, label %140

110:                                              ; preds = %108
  %111 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %112 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %111, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %113 = zext i32 %25 to i64
  %114 = mul i64 %98, %113
  %115 = zext i32 %85 to i64
  %116 = add i64 %114, %115
  br label %117

117:                                              ; preds = %125, %110
  %118 = phi i32 [ 0, %110 ], [ %126, %125 ]
  %119 = add nuw nsw i32 %118, %85
  %120 = icmp ult i32 %119, %25
  br i1 %120, label %121, label %125

121:                                              ; preds = %117
  %122 = zext i32 %118 to i64
  %123 = add i64 %116, %122
  %124 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %123
  store bfloat 0xR7FC0, bfloat addrspace(1)* %124, align 2, !tbaa !56
  br label %125

125:                                              ; preds = %121, %117
  %126 = add nuw nsw i32 %118, 1
  %127 = icmp eq i32 %126, 4
  br i1 %127, label %140, label %117, !llvm.loop !65

128:                                              ; preds = %102
  %129 = icmp eq i32 %47, 0
  %130 = select i1 %129, i64 %94, i64 %98
  %131 = mul i64 %130, %59
  %132 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %131
  %133 = icmp ugt i32 %25, 7
  %134 = and i32 %29, 255
  %135 = icmp eq i32 %134, 0
  %136 = select i1 %133, i1 %135, i1 false
  %137 = trunc i64 %103 to i32
  br i1 %136, label %138, label %139

138:                                              ; preds = %128
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %132, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %85, i32 noundef %137, i64 noundef %98, i32 noundef %10) #10
  br label %140

139:                                              ; preds = %128
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %132, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %85, i32 noundef %137, i64 noundef %98, i32 noundef %10) #10
  br label %140

140:                                              ; preds = %139, %138, %125, %108, %90, %87, %81, %78, %76
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @flash_affine_mlx_qmv_f32xsum_v1_q8_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #0 {
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt8ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) #10
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v17projectILt8ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %13 = load i32, i32 addrspace(2)* %12, align 8, !tbaa !39
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %76, label %15

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %17 = load i32, i32 addrspace(2)* %16, align 4, !tbaa !45
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %76, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %21 = load i32, i32 addrspace(2)* %20, align 8, !tbaa !46
  %22 = icmp eq i32 %21, 0
  br i1 %22, label %76, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %76, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %29 = load i32, i32 addrspace(2)* %28, align 8, !tbaa !48
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %76, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !49
  %34 = icmp eq i32 %33, 8
  br i1 %34, label %35, label %76

35:                                               ; preds = %31
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %37 = load i32, i32 addrspace(2)* %36, align 8, !tbaa !50
  %38 = icmp eq i32 %37, 128
  %39 = and i32 %29, 127
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 %40, i1 false
  br i1 %41, label %42, label %76

42:                                               ; preds = %35
  %43 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %44 = load i32, i32 addrspace(2)* %43, align 4, !tbaa !51
  %45 = icmp ult i32 %44, 4
  br i1 %45, label %46, label %76

46:                                               ; preds = %42
  %47 = and i32 %44, 2
  %48 = and i32 %44, 1
  %49 = icmp eq i32 %48, 0
  %50 = icmp eq i32 %44, 2
  br i1 %50, label %76, label %51

51:                                               ; preds = %46
  br i1 %49, label %52, label %56

52:                                               ; preds = %51
  %53 = icmp eq i32 %21, 1
  %54 = icmp eq i32 %17, 1
  %55 = select i1 %53, i1 %54, i1 false
  br i1 %55, label %56, label %76

56:                                               ; preds = %52, %51
  %57 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %58 = load i64, i64 addrspace(2)* %57, align 8, !tbaa !52
  %59 = zext i32 %29 to i64
  %60 = icmp ult i64 %58, %59
  br i1 %60, label %76, label %61

61:                                               ; preds = %56
  %62 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %63 = load i64, i64 addrspace(2)* %62, align 8, !tbaa !53
  %64 = lshr i32 %29, 6
  %65 = and i32 %64, 67108862
  %66 = zext i32 %65 to i64
  %67 = icmp uge i64 %63, %66
  %68 = and i64 %63, 1
  %69 = icmp eq i64 %68, 0
  %70 = and i1 %67, %69
  br i1 %70, label %71, label %76

71:                                               ; preds = %61
  %72 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %73 = load i64, i64 addrspace(2)* %72, align 8, !tbaa !54
  %74 = and i64 %73, 1
  %75 = icmp eq i64 %74, 0
  br i1 %75, label %81, label %76

76:                                               ; preds = %71, %61, %56, %52, %46, %42, %35, %31, %27, %23, %19, %15, %11
  %77 = icmp eq i32 %10, 0
  br i1 %77, label %78, label %140

78:                                               ; preds = %76
  %79 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %80 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %79, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %140

81:                                               ; preds = %71
  %82 = extractelement <3 x i32> %8, i64 0
  %83 = shl i32 %82, 3
  %84 = shl i32 %9, 2
  %85 = add i32 %83, %84
  %86 = icmp ult i32 %85, %25
  br i1 %86, label %87, label %140

87:                                               ; preds = %81
  %88 = extractelement <3 x i32> %8, i64 1
  %89 = icmp ult i32 %88, %13
  br i1 %89, label %90, label %140

90:                                               ; preds = %87
  %91 = extractelement <3 x i32> %8, i64 2
  %92 = icmp ult i32 %91, %17
  br i1 %92, label %93, label %140

93:                                               ; preds = %90
  %94 = zext i32 %88 to i64
  %95 = zext i32 %17 to i64
  %96 = mul nuw i64 %95, %94
  %97 = zext i32 %91 to i64
  %98 = add nuw i64 %96, %97
  br i1 %49, label %102, label %99

99:                                               ; preds = %93
  %100 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %98
  %101 = load i64, i64 addrspace(1)* %100, align 8, !tbaa !55
  br label %102

102:                                              ; preds = %99, %93
  %103 = phi i64 [ %101, %99 ], [ 0, %93 ]
  %104 = icmp sgt i64 %103, -1
  %105 = zext i32 %21 to i64
  %106 = icmp ult i64 %103, %105
  %107 = select i1 %104, i1 %106, i1 false
  br i1 %107, label %128, label %108

108:                                              ; preds = %102
  %109 = icmp eq i32 %10, 0
  br i1 %109, label %110, label %140

110:                                              ; preds = %108
  %111 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %112 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %111, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %113 = zext i32 %25 to i64
  %114 = mul i64 %98, %113
  %115 = zext i32 %85 to i64
  %116 = add i64 %114, %115
  br label %117

117:                                              ; preds = %125, %110
  %118 = phi i32 [ 0, %110 ], [ %126, %125 ]
  %119 = add nuw nsw i32 %118, %85
  %120 = icmp ult i32 %119, %25
  br i1 %120, label %121, label %125

121:                                              ; preds = %117
  %122 = zext i32 %118 to i64
  %123 = add i64 %116, %122
  %124 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %123
  store bfloat 0xR7FC0, bfloat addrspace(1)* %124, align 2, !tbaa !56
  br label %125

125:                                              ; preds = %121, %117
  %126 = add nuw nsw i32 %118, 1
  %127 = icmp eq i32 %126, 4
  br i1 %127, label %140, label %117, !llvm.loop !66

128:                                              ; preds = %102
  %129 = icmp eq i32 %47, 0
  %130 = select i1 %129, i64 %94, i64 %98
  %131 = mul i64 %130, %59
  %132 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %131
  %133 = icmp ugt i32 %25, 7
  %134 = and i32 %29, 255
  %135 = icmp eq i32 %134, 0
  %136 = select i1 %133, i1 %135, i1 false
  %137 = trunc i64 %103 to i32
  br i1 %136, label %138, label %139

138:                                              ; preds = %128
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %132, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %85, i32 noundef %137, i64 noundef %98, i32 noundef %10) #10
  br label %140

139:                                              ; preds = %128
  tail call void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %132, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %85, i32 noundef %137, i64 noundef %98, i32 noundef %10) #10
  br label %140

140:                                              ; preds = %139, %138, %125, %108, %90, %87, %81, %78, %76
  ret void
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %23, label %19

19:                                               ; preds = %11
  %20 = shl i32 %10, 4
  %21 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %32

23:                                               ; preds = %72, %11
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %10, 0
  %27 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %28 = zext i32 %25 to i64
  %29 = mul i64 %28, %9
  %30 = zext i32 %7 to i64
  %31 = add i64 %29, %30
  br label %76

32:                                               ; preds = %72, %19
  %33 = phi i32 [ 0, %19 ], [ %73, %72 ]
  %34 = add i32 %33, %20
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %35
  br label %37

37:                                               ; preds = %37, %32
  %38 = phi i32 [ 0, %32 ], [ %70, %37 ]
  %39 = phi float [ 0.000000e+00, %32 ], [ %62, %37 ]
  %40 = zext i32 %38 to i64
  %41 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %40
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !56
  %43 = fpext bfloat %42 to float
  %44 = or i32 %38, 1
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %45
  %47 = load bfloat, bfloat addrspace(1)* %46, align 2, !tbaa !56
  %48 = fpext bfloat %47 to float
  %49 = fadd float %43, %48
  %50 = or i32 %38, 2
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %51
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !56
  %54 = fpext bfloat %53 to float
  %55 = fadd float %49, %54
  %56 = or i32 %38, 3
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %57
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !56
  %60 = fpext bfloat %59 to float
  %61 = fadd float %55, %60
  %62 = fadd float %39, %61
  %63 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %40
  store float %43, float* %63, align 4, !tbaa !67
  %64 = fmul float %48, 6.250000e-02
  %65 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %45
  store float %64, float* %65, align 4, !tbaa !67
  %66 = fmul float %54, 3.906250e-03
  %67 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %51
  store float %66, float* %67, align 4, !tbaa !67
  %68 = fmul float %60, 0x3F30000000000000
  %69 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %57
  store float %68, float* %69, align 4, !tbaa !67
  %70 = add nuw nsw i32 %38, 4
  %71 = icmp ult i32 %38, 12
  br i1 %71, label %37, label %72, !llvm.loop !69

72:                                               ; preds = %37
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %34, i32 noundef %8, float* noundef nonnull %21, float noundef %62, i32 noundef 16, i1 noundef zeroext false, float* noundef nonnull %22) #13
  %73 = add i32 %33, 512
  %74 = icmp ult i32 %73, %17
  br i1 %74, label %32, label %23, !llvm.loop !70

75:                                               ; preds = %100
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #12
  ret void

76:                                               ; preds = %100, %23
  %77 = phi i32 [ 0, %23 ], [ %101, %100 ]
  %78 = add i32 %77, %7
  %79 = icmp ult i32 %78, %25
  br i1 %79, label %80, label %100

80:                                               ; preds = %76
  %81 = zext i32 %77 to i64
  %82 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %81
  %83 = load float, float* %82, align 4, !tbaa !67
  %84 = call fast float @air.simd_sum.f32(float %83) #14
  br i1 %26, label %85, label %100

85:                                               ; preds = %80
  %86 = fptrunc float %84 to bfloat
  %87 = bitcast float %84 to i32
  %88 = and i32 %87, 2139095040
  %89 = icmp eq i32 %88, 2139095040
  br i1 %89, label %95, label %90

90:                                               ; preds = %85
  %91 = fpext bfloat %86 to float
  %92 = bitcast float %91 to i32
  %93 = and i32 %92, 2139095040
  %94 = icmp eq i32 %93, 2139095040
  br i1 %94, label %95, label %97

95:                                               ; preds = %90, %85
  %96 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %27, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %97

97:                                               ; preds = %95, %90
  %98 = add i64 %31, %81
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %98
  store bfloat %86, bfloat addrspace(1)* %99, align 2, !tbaa !56
  br label %100

100:                                              ; preds = %97, %80, %76
  %101 = add nuw nsw i32 %77, 1
  %102 = icmp eq i32 %101, 4
  br i1 %102, label %75, label %76, !llvm.loop !71
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 256
  %19 = shl i32 %10, 3
  br i1 %18, label %20, label %68

20:                                               ; preds = %11
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %23

23:                                               ; preds = %62, %20
  %24 = phi i32 [ 0, %20 ], [ %63, %62 ]
  %25 = add i32 %24, %19
  %26 = zext i32 %25 to i64
  %27 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %26
  br label %28

28:                                               ; preds = %28, %23
  %29 = phi i1 [ true, %23 ], [ false, %28 ]
  %30 = phi i32 [ 0, %23 ], [ 4, %28 ]
  %31 = phi float [ 0.000000e+00, %23 ], [ %54, %28 ]
  %32 = zext i32 %30 to i64
  %33 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %32
  %34 = load bfloat, bfloat addrspace(1)* %33, align 2, !tbaa !56
  %35 = fpext bfloat %34 to float
  %36 = or i32 %30, 1
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %37
  %39 = load bfloat, bfloat addrspace(1)* %38, align 2, !tbaa !56
  %40 = fpext bfloat %39 to float
  %41 = fadd float %35, %40
  %42 = or i32 %30, 2
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !56
  %46 = fpext bfloat %45 to float
  %47 = fadd float %41, %46
  %48 = or i32 %30, 3
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %49
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !56
  %52 = fpext bfloat %51 to float
  %53 = fadd float %47, %52
  %54 = fadd float %31, %53
  %55 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %32
  store float %35, float* %55, align 4, !tbaa !67
  %56 = fmul float %40, 6.250000e-02
  %57 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %37
  store float %56, float* %57, align 4, !tbaa !67
  %58 = fmul float %46, 3.906250e-03
  %59 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %43
  store float %58, float* %59, align 4, !tbaa !67
  %60 = fmul float %52, 0x3F30000000000000
  %61 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %49
  store float %60, float* %61, align 4, !tbaa !67
  br i1 %29, label %28, label %62, !llvm.loop !72

62:                                               ; preds = %28
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %25, i32 noundef %8, float* noundef nonnull %21, float noundef %54, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %22) #13
  %63 = add i32 %24, 256
  %64 = icmp ugt i32 %17, %63
  %65 = sub i32 %17, %63
  %66 = icmp ugt i32 %65, 256
  %67 = and i1 %64, %66
  br i1 %67, label %23, label %68, !llvm.loop !73

68:                                               ; preds = %62, %11
  %69 = phi i32 [ 0, %11 ], [ %63, %62 ]
  %70 = phi i32 [ %17, %11 ], [ %65, %62 ]
  %71 = icmp ugt i32 %70, %19
  br i1 %71, label %72, label %75

72:                                               ; preds = %68
  %73 = sub i32 %70, %19
  %74 = call i32 @air.min.u.i32(i32 %73, i32 8) #15
  br label %75

75:                                               ; preds = %72, %68
  %76 = phi i32 [ %74, %72 ], [ 0, %68 ]
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %130, label %78

78:                                               ; preds = %75
  %79 = add i32 %69, %19
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %80
  %82 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %83 = icmp sgt i32 %76, 0
  br i1 %83, label %87, label %84

84:                                               ; preds = %87, %78
  %85 = phi float [ 0.000000e+00, %78 ], [ %112, %87 ]
  %86 = icmp slt i32 %76, 8
  br i1 %86, label %122, label %128

87:                                               ; preds = %87, %78
  %88 = phi i32 [ %120, %87 ], [ 0, %78 ]
  %89 = phi float [ %112, %87 ], [ 0.000000e+00, %78 ]
  %90 = zext i32 %88 to i64
  %91 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %90
  %92 = load bfloat, bfloat addrspace(1)* %91, align 2, !tbaa !56
  %93 = fpext bfloat %92 to float
  %94 = or i32 %88, 1
  %95 = zext i32 %94 to i64
  %96 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %95
  %97 = load bfloat, bfloat addrspace(1)* %96, align 2, !tbaa !56
  %98 = fpext bfloat %97 to float
  %99 = fadd float %93, %98
  %100 = or i32 %88, 2
  %101 = zext i32 %100 to i64
  %102 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %101
  %103 = load bfloat, bfloat addrspace(1)* %102, align 2, !tbaa !56
  %104 = fpext bfloat %103 to float
  %105 = fadd float %99, %104
  %106 = or i32 %88, 3
  %107 = zext i32 %106 to i64
  %108 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %107
  %109 = load bfloat, bfloat addrspace(1)* %108, align 2, !tbaa !56
  %110 = fpext bfloat %109 to float
  %111 = fadd float %105, %110
  %112 = fadd float %89, %111
  %113 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %90
  store float %93, float* %113, align 4, !tbaa !67
  %114 = fmul float %98, 6.250000e-02
  %115 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %95
  store float %114, float* %115, align 4, !tbaa !67
  %116 = fmul float %104, 3.906250e-03
  %117 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %101
  store float %116, float* %117, align 4, !tbaa !67
  %118 = fmul float %110, 0x3F30000000000000
  %119 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %107
  store float %118, float* %119, align 4, !tbaa !67
  %120 = add nuw nsw i32 %88, 4
  %121 = icmp slt i32 %120, %76
  br i1 %121, label %87, label %84, !llvm.loop !74

122:                                              ; preds = %122, %84
  %123 = phi i32 [ %126, %122 ], [ %76, %84 ]
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %124
  store float 0.000000e+00, float* %125, align 4, !tbaa !67
  %126 = add i32 %123, 1
  %127 = icmp eq i32 %126, 8
  br i1 %127, label %128, label %122, !llvm.loop !75

128:                                              ; preds = %122, %84
  %129 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %79, i32 noundef %8, float* noundef nonnull %82, float noundef %85, i32 noundef %76, i1 noundef zeroext true, float* noundef nonnull %129) #13
  br label %130

130:                                              ; preds = %128, %75
  %131 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %132 = load i32, i32 addrspace(2)* %131, align 4, !tbaa !47
  %133 = icmp eq i32 %10, 0
  %134 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %135 = zext i32 %132 to i64
  %136 = mul i64 %135, %9
  %137 = zext i32 %7 to i64
  %138 = add i64 %136, %137
  br label %140

139:                                              ; preds = %164
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

140:                                              ; preds = %164, %130
  %141 = phi i32 [ 0, %130 ], [ %165, %164 ]
  %142 = add i32 %141, %7
  %143 = icmp ult i32 %142, %132
  br i1 %143, label %144, label %164

144:                                              ; preds = %140
  %145 = zext i32 %141 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !67
  %148 = call fast float @air.simd_sum.f32(float %147) #14
  br i1 %133, label %149, label %164

149:                                              ; preds = %144
  %150 = fptrunc float %148 to bfloat
  %151 = bitcast float %148 to i32
  %152 = and i32 %151, 2139095040
  %153 = icmp eq i32 %152, 2139095040
  br i1 %153, label %159, label %154

154:                                              ; preds = %149
  %155 = fpext bfloat %150 to float
  %156 = bitcast float %155 to i32
  %157 = and i32 %156, 2139095040
  %158 = icmp eq i32 %157, 2139095040
  br i1 %158, label %159, label %161

159:                                              ; preds = %154, %149
  %160 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %134, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %161

161:                                              ; preds = %159, %154
  %162 = add i64 %138, %145
  %163 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %162
  store bfloat %150, bfloat addrspace(1)* %163, align 2, !tbaa !56
  br label %164

164:                                              ; preds = %161, %144, %140
  %165 = add nuw nsw i32 %141, 1
  %166 = icmp eq i32 %165, 4
  br i1 %166, label %139, label %140, !llvm.loop !76
}

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture, i32, i32, i32, i32, i1) local_unnamed_addr #4

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #5

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #6 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = lshr i32 %5, 1
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %29 = load i64, i64 addrspace(2)* %28, align 8
  %30 = zext i32 %20 to i64
  %31 = sdiv i32 %9, 4
  %32 = icmp sgt i32 %9, 3
  %33 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %27
  br label %35

34:                                               ; preds = %161
  ret void

35:                                               ; preds = %161, %12
  %36 = phi i32 [ 0, %12 ], [ %162, %161 ]
  %37 = add i32 %36, %4
  %38 = icmp ult i32 %37, %22
  br i1 %38, label %39, label %161

39:                                               ; preds = %35
  %40 = zext i32 %37 to i64
  %41 = mul i64 %25, %40
  %42 = getelementptr inbounds i8, i8 addrspace(1)* %33, i64 %41
  %43 = mul i64 %29, %40
  %44 = add i64 %43, %16
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %44
  %46 = bitcast i8 addrspace(1)* %45 to bfloat addrspace(1)*
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %44
  %48 = bitcast i8 addrspace(1)* %47 to bfloat addrspace(1)*
  %49 = getelementptr inbounds bfloat, bfloat addrspace(1)* %46, i64 %30
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !56
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !56
  %54 = fpext bfloat %53 to float
  br i1 %10, label %55, label %109

55:                                               ; preds = %39
  br i1 %32, label %56, label %101

56:                                               ; preds = %56, %55
  %57 = phi float [ %98, %56 ], [ 0.000000e+00, %55 ]
  %58 = phi i32 [ %99, %56 ], [ 0, %55 ]
  %59 = shl nuw nsw i32 %58, 1
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %60
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !78
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !78
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !67
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #15
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !67
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #15
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #12
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !67
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #15
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #12
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !67
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #15
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #12
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !79

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #12
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !67
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !67
  br label %161

109:                                              ; preds = %109, %39
  %110 = phi float [ %151, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i32 [ %152, %109 ], [ 0, %39 ]
  %112 = shl nuw nsw i32 %111, 1
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %113
  %115 = load i8, i8 addrspace(1)* %114, align 1, !tbaa !78
  %116 = zext i8 %115 to i32
  %117 = or i32 %112, 1
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %118
  %120 = load i8, i8 addrspace(1)* %119, align 1, !tbaa !78
  %121 = zext i8 %120 to i32
  %122 = shl nuw nsw i32 %121, 8
  %123 = shl nuw nsw i32 %111, 2
  %124 = zext i32 %123 to i64
  %125 = getelementptr inbounds float, float* %7, i64 %124
  %126 = load float, float* %125, align 4, !tbaa !67
  %127 = and i32 %116, 15
  %128 = tail call float @air.convert.f.f32.s.i32(i32 %127) #15
  %129 = or i32 %123, 1
  %130 = zext i32 %129 to i64
  %131 = getelementptr inbounds float, float* %7, i64 %130
  %132 = load float, float* %131, align 4, !tbaa !67
  %133 = and i32 %116, 240
  %134 = tail call float @air.convert.f.f32.s.i32(i32 %133) #15
  %135 = fmul float %132, %134
  %136 = tail call float @llvm.fmuladd.f32(float %126, float %128, float %135) #12
  %137 = or i32 %123, 2
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds float, float* %7, i64 %138
  %140 = load float, float* %139, align 4, !tbaa !67
  %141 = and i32 %122, 3840
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #15
  %143 = tail call float @llvm.fmuladd.f32(float %140, float %142, float %136) #12
  %144 = or i32 %123, 3
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds float, float* %7, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !67
  %148 = and i32 %122, 61440
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #15
  %150 = tail call float @llvm.fmuladd.f32(float %147, float %149, float %143) #12
  %151 = fadd float %110, %150
  %152 = add nuw nsw i32 %111, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %154, label %109, !llvm.loop !80

154:                                              ; preds = %109
  %155 = fmul float %54, %8
  %156 = tail call float @llvm.fmuladd.f32(float %51, float %151, float %155) #12
  %157 = zext i32 %36 to i64
  %158 = getelementptr inbounds float, float* %11, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !67
  %160 = fadd float %159, %156
  store float %160, float* %158, align 4, !tbaa !67
  br label %161

161:                                              ; preds = %154, %101, %35
  %162 = add nuw nsw i32 %36, 1
  %163 = icmp eq i32 %162, 4
  br i1 %163, label %34, label %35, !llvm.loop !81
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #7

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #8

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #9

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #6 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = lshr i32 %5, 1
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %29 = load i64, i64 addrspace(2)* %28, align 8
  %30 = zext i32 %20 to i64
  %31 = sdiv i32 %9, 4
  %32 = icmp sgt i32 %9, 3
  %33 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %27
  br label %35

34:                                               ; preds = %160
  ret void

35:                                               ; preds = %160, %12
  %36 = phi i32 [ 0, %12 ], [ %161, %160 ]
  %37 = add i32 %36, %4
  %38 = icmp ult i32 %37, %22
  br i1 %38, label %39, label %160

39:                                               ; preds = %35
  %40 = zext i32 %37 to i64
  %41 = mul i64 %25, %40
  %42 = getelementptr inbounds i8, i8 addrspace(1)* %33, i64 %41
  %43 = mul i64 %29, %40
  %44 = add i64 %43, %16
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %44
  %46 = bitcast i8 addrspace(1)* %45 to bfloat addrspace(1)*
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %44
  %48 = bitcast i8 addrspace(1)* %47 to bfloat addrspace(1)*
  %49 = getelementptr inbounds bfloat, bfloat addrspace(1)* %46, i64 %30
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !56
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !56
  %54 = fpext bfloat %53 to float
  br i1 %10, label %55, label %109

55:                                               ; preds = %39
  br i1 %32, label %56, label %101

56:                                               ; preds = %56, %55
  %57 = phi float [ %98, %56 ], [ 0.000000e+00, %55 ]
  %58 = phi i32 [ %99, %56 ], [ 0, %55 ]
  %59 = shl nuw nsw i32 %58, 1
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %60
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !78
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !78
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !67
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #15
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !67
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #15
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #12
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !67
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #15
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #12
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !67
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #15
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #12
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !82

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #12
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !67
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !67
  br label %160

109:                                              ; preds = %109, %39
  %110 = phi float [ %152, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i1 [ false, %109 ], [ true, %39 ]
  %112 = phi i32 [ 1, %109 ], [ 0, %39 ]
  %113 = shl nuw nsw i32 %112, 1
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %114
  %116 = load i8, i8 addrspace(1)* %115, align 1, !tbaa !78
  %117 = zext i8 %116 to i32
  %118 = or i32 %113, 1
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %119
  %121 = load i8, i8 addrspace(1)* %120, align 1, !tbaa !78
  %122 = zext i8 %121 to i32
  %123 = shl nuw nsw i32 %122, 8
  %124 = shl nuw nsw i32 %112, 2
  %125 = zext i32 %124 to i64
  %126 = getelementptr inbounds float, float* %7, i64 %125
  %127 = load float, float* %126, align 4, !tbaa !67
  %128 = and i32 %117, 15
  %129 = tail call float @air.convert.f.f32.s.i32(i32 %128) #15
  %130 = or i32 %124, 1
  %131 = zext i32 %130 to i64
  %132 = getelementptr inbounds float, float* %7, i64 %131
  %133 = load float, float* %132, align 4, !tbaa !67
  %134 = and i32 %117, 240
  %135 = tail call float @air.convert.f.f32.s.i32(i32 %134) #15
  %136 = fmul float %133, %135
  %137 = tail call float @llvm.fmuladd.f32(float %127, float %129, float %136) #12
  %138 = or i32 %124, 2
  %139 = zext i32 %138 to i64
  %140 = getelementptr inbounds float, float* %7, i64 %139
  %141 = load float, float* %140, align 4, !tbaa !67
  %142 = and i32 %123, 3840
  %143 = tail call float @air.convert.f.f32.s.i32(i32 %142) #15
  %144 = tail call float @llvm.fmuladd.f32(float %141, float %143, float %137) #12
  %145 = or i32 %124, 3
  %146 = zext i32 %145 to i64
  %147 = getelementptr inbounds float, float* %7, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !67
  %149 = and i32 %123, 61440
  %150 = tail call float @air.convert.f.f32.s.i32(i32 %149) #15
  %151 = tail call float @llvm.fmuladd.f32(float %148, float %150, float %144) #12
  %152 = fadd float %110, %151
  br i1 %111, label %109, label %153, !llvm.loop !83

153:                                              ; preds = %109
  %154 = fmul float %54, %8
  %155 = tail call float @llvm.fmuladd.f32(float %51, float %152, float %154) #12
  %156 = zext i32 %36 to i64
  %157 = getelementptr inbounds float, float* %11, i64 %156
  %158 = load float, float* %157, align 4, !tbaa !67
  %159 = fadd float %158, %155
  store float %159, float* %157, align 4, !tbaa !67
  br label %160

160:                                              ; preds = %153, %101, %35
  %161 = add nuw nsw i32 %36, 1
  %162 = icmp eq i32 %161, 4
  br i1 %162, label %34, label %35, !llvm.loop !84
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @air.min.u.i32(i32, i32) local_unnamed_addr #7

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %23, label %19

19:                                               ; preds = %11
  %20 = shl i32 %10, 4
  %21 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %32

23:                                               ; preds = %72, %11
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %10, 0
  %27 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %28 = zext i32 %25 to i64
  %29 = mul i64 %28, %9
  %30 = zext i32 %7 to i64
  %31 = add i64 %29, %30
  br label %76

32:                                               ; preds = %72, %19
  %33 = phi i32 [ 0, %19 ], [ %73, %72 ]
  %34 = add i32 %33, %20
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %35
  br label %37

37:                                               ; preds = %37, %32
  %38 = phi i32 [ 0, %32 ], [ %70, %37 ]
  %39 = phi float [ 0.000000e+00, %32 ], [ %62, %37 ]
  %40 = zext i32 %38 to i64
  %41 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %40
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !56
  %43 = fpext bfloat %42 to float
  %44 = or i32 %38, 1
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %45
  %47 = load bfloat, bfloat addrspace(1)* %46, align 2, !tbaa !56
  %48 = fpext bfloat %47 to float
  %49 = fadd float %43, %48
  %50 = or i32 %38, 2
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %51
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !56
  %54 = fpext bfloat %53 to float
  %55 = fadd float %49, %54
  %56 = or i32 %38, 3
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %57
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !56
  %60 = fpext bfloat %59 to float
  %61 = fadd float %55, %60
  %62 = fadd float %39, %61
  %63 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %40
  store float %43, float* %63, align 4, !tbaa !67
  %64 = fmul float %48, 6.250000e-02
  %65 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %45
  store float %64, float* %65, align 4, !tbaa !67
  %66 = fmul float %54, 3.906250e-03
  %67 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %51
  store float %66, float* %67, align 4, !tbaa !67
  %68 = fmul float %60, 0x3F30000000000000
  %69 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %57
  store float %68, float* %69, align 4, !tbaa !67
  %70 = add nuw nsw i32 %38, 4
  %71 = icmp ult i32 %38, 12
  br i1 %71, label %37, label %72, !llvm.loop !69

72:                                               ; preds = %37
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt128ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %34, i32 noundef %8, float* noundef nonnull %21, float noundef %62, i32 noundef 16, i1 noundef zeroext false, float* noundef nonnull %22) #13
  %73 = add i32 %33, 512
  %74 = icmp ult i32 %73, %17
  br i1 %74, label %32, label %23, !llvm.loop !85

75:                                               ; preds = %100
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #12
  ret void

76:                                               ; preds = %100, %23
  %77 = phi i32 [ 0, %23 ], [ %101, %100 ]
  %78 = add i32 %77, %7
  %79 = icmp ult i32 %78, %25
  br i1 %79, label %80, label %100

80:                                               ; preds = %76
  %81 = zext i32 %77 to i64
  %82 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %81
  %83 = load float, float* %82, align 4, !tbaa !67
  %84 = call fast float @air.simd_sum.f32(float %83) #14
  br i1 %26, label %85, label %100

85:                                               ; preds = %80
  %86 = fptrunc float %84 to bfloat
  %87 = bitcast float %84 to i32
  %88 = and i32 %87, 2139095040
  %89 = icmp eq i32 %88, 2139095040
  br i1 %89, label %95, label %90

90:                                               ; preds = %85
  %91 = fpext bfloat %86 to float
  %92 = bitcast float %91 to i32
  %93 = and i32 %92, 2139095040
  %94 = icmp eq i32 %93, 2139095040
  br i1 %94, label %95, label %97

95:                                               ; preds = %90, %85
  %96 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %27, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %97

97:                                               ; preds = %95, %90
  %98 = add i64 %31, %81
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %98
  store bfloat %86, bfloat addrspace(1)* %99, align 2, !tbaa !56
  br label %100

100:                                              ; preds = %97, %80, %76
  %101 = add nuw nsw i32 %77, 1
  %102 = icmp eq i32 %101, 4
  br i1 %102, label %75, label %76, !llvm.loop !86
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt4ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 256
  %19 = shl i32 %10, 3
  br i1 %18, label %20, label %68

20:                                               ; preds = %11
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %23

23:                                               ; preds = %62, %20
  %24 = phi i32 [ 0, %20 ], [ %63, %62 ]
  %25 = add i32 %24, %19
  %26 = zext i32 %25 to i64
  %27 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %26
  br label %28

28:                                               ; preds = %28, %23
  %29 = phi i1 [ true, %23 ], [ false, %28 ]
  %30 = phi i32 [ 0, %23 ], [ 4, %28 ]
  %31 = phi float [ 0.000000e+00, %23 ], [ %54, %28 ]
  %32 = zext i32 %30 to i64
  %33 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %32
  %34 = load bfloat, bfloat addrspace(1)* %33, align 2, !tbaa !56
  %35 = fpext bfloat %34 to float
  %36 = or i32 %30, 1
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %37
  %39 = load bfloat, bfloat addrspace(1)* %38, align 2, !tbaa !56
  %40 = fpext bfloat %39 to float
  %41 = fadd float %35, %40
  %42 = or i32 %30, 2
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !56
  %46 = fpext bfloat %45 to float
  %47 = fadd float %41, %46
  %48 = or i32 %30, 3
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %27, i64 %49
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !56
  %52 = fpext bfloat %51 to float
  %53 = fadd float %47, %52
  %54 = fadd float %31, %53
  %55 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %32
  store float %35, float* %55, align 4, !tbaa !67
  %56 = fmul float %40, 6.250000e-02
  %57 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %37
  store float %56, float* %57, align 4, !tbaa !67
  %58 = fmul float %46, 3.906250e-03
  %59 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %43
  store float %58, float* %59, align 4, !tbaa !67
  %60 = fmul float %52, 0x3F30000000000000
  %61 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %49
  store float %60, float* %61, align 4, !tbaa !67
  br i1 %29, label %28, label %62, !llvm.loop !72

62:                                               ; preds = %28
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt128ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %25, i32 noundef %8, float* noundef nonnull %21, float noundef %54, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %22) #13
  %63 = add i32 %24, 256
  %64 = icmp ugt i32 %17, %63
  %65 = sub i32 %17, %63
  %66 = icmp ugt i32 %65, 256
  %67 = and i1 %64, %66
  br i1 %67, label %23, label %68, !llvm.loop !87

68:                                               ; preds = %62, %11
  %69 = phi i32 [ 0, %11 ], [ %63, %62 ]
  %70 = phi i32 [ %17, %11 ], [ %65, %62 ]
  %71 = icmp ugt i32 %70, %19
  br i1 %71, label %72, label %75

72:                                               ; preds = %68
  %73 = sub i32 %70, %19
  %74 = call i32 @air.min.u.i32(i32 %73, i32 8) #15
  br label %75

75:                                               ; preds = %72, %68
  %76 = phi i32 [ %74, %72 ], [ 0, %68 ]
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %130, label %78

78:                                               ; preds = %75
  %79 = add i32 %69, %19
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %80
  %82 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %83 = icmp sgt i32 %76, 0
  br i1 %83, label %87, label %84

84:                                               ; preds = %87, %78
  %85 = phi float [ 0.000000e+00, %78 ], [ %112, %87 ]
  %86 = icmp slt i32 %76, 8
  br i1 %86, label %122, label %128

87:                                               ; preds = %87, %78
  %88 = phi i32 [ %120, %87 ], [ 0, %78 ]
  %89 = phi float [ %112, %87 ], [ 0.000000e+00, %78 ]
  %90 = zext i32 %88 to i64
  %91 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %90
  %92 = load bfloat, bfloat addrspace(1)* %91, align 2, !tbaa !56
  %93 = fpext bfloat %92 to float
  %94 = or i32 %88, 1
  %95 = zext i32 %94 to i64
  %96 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %95
  %97 = load bfloat, bfloat addrspace(1)* %96, align 2, !tbaa !56
  %98 = fpext bfloat %97 to float
  %99 = fadd float %93, %98
  %100 = or i32 %88, 2
  %101 = zext i32 %100 to i64
  %102 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %101
  %103 = load bfloat, bfloat addrspace(1)* %102, align 2, !tbaa !56
  %104 = fpext bfloat %103 to float
  %105 = fadd float %99, %104
  %106 = or i32 %88, 3
  %107 = zext i32 %106 to i64
  %108 = getelementptr inbounds bfloat, bfloat addrspace(1)* %81, i64 %107
  %109 = load bfloat, bfloat addrspace(1)* %108, align 2, !tbaa !56
  %110 = fpext bfloat %109 to float
  %111 = fadd float %105, %110
  %112 = fadd float %89, %111
  %113 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %90
  store float %93, float* %113, align 4, !tbaa !67
  %114 = fmul float %98, 6.250000e-02
  %115 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %95
  store float %114, float* %115, align 4, !tbaa !67
  %116 = fmul float %104, 3.906250e-03
  %117 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %101
  store float %116, float* %117, align 4, !tbaa !67
  %118 = fmul float %110, 0x3F30000000000000
  %119 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %107
  store float %118, float* %119, align 4, !tbaa !67
  %120 = add nuw nsw i32 %88, 4
  %121 = icmp slt i32 %120, %76
  br i1 %121, label %87, label %84, !llvm.loop !74

122:                                              ; preds = %122, %84
  %123 = phi i32 [ %126, %122 ], [ %76, %84 ]
  %124 = sext i32 %123 to i64
  %125 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %124
  store float 0.000000e+00, float* %125, align 4, !tbaa !67
  %126 = add i32 %123, 1
  %127 = icmp eq i32 %126, 8
  br i1 %127, label %128, label %122, !llvm.loop !75

128:                                              ; preds = %122, %84
  %129 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt128ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %79, i32 noundef %8, float* noundef nonnull %82, float noundef %85, i32 noundef %76, i1 noundef zeroext true, float* noundef nonnull %129) #13
  br label %130

130:                                              ; preds = %128, %75
  %131 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %132 = load i32, i32 addrspace(2)* %131, align 4, !tbaa !47
  %133 = icmp eq i32 %10, 0
  %134 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %135 = zext i32 %132 to i64
  %136 = mul i64 %135, %9
  %137 = zext i32 %7 to i64
  %138 = add i64 %136, %137
  br label %140

139:                                              ; preds = %164
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

140:                                              ; preds = %164, %130
  %141 = phi i32 [ 0, %130 ], [ %165, %164 ]
  %142 = add i32 %141, %7
  %143 = icmp ult i32 %142, %132
  br i1 %143, label %144, label %164

144:                                              ; preds = %140
  %145 = zext i32 %141 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !67
  %148 = call fast float @air.simd_sum.f32(float %147) #14
  br i1 %133, label %149, label %164

149:                                              ; preds = %144
  %150 = fptrunc float %148 to bfloat
  %151 = bitcast float %148 to i32
  %152 = and i32 %151, 2139095040
  %153 = icmp eq i32 %152, 2139095040
  br i1 %153, label %159, label %154

154:                                              ; preds = %149
  %155 = fpext bfloat %150 to float
  %156 = bitcast float %155 to i32
  %157 = and i32 %156, 2139095040
  %158 = icmp eq i32 %157, 2139095040
  br i1 %158, label %159, label %161

159:                                              ; preds = %154, %149
  %160 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %134, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %161

161:                                              ; preds = %159, %154
  %162 = add i64 %138, %145
  %163 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %162
  store bfloat %150, bfloat addrspace(1)* %163, align 2, !tbaa !56
  br label %164

164:                                              ; preds = %161, %144, %140
  %165 = add nuw nsw i32 %141, 1
  %166 = icmp eq i32 %165, 4
  br i1 %166, label %139, label %140, !llvm.loop !88
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt128ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #6 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 7
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = lshr i32 %5, 1
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %29 = load i64, i64 addrspace(2)* %28, align 8
  %30 = zext i32 %20 to i64
  %31 = sdiv i32 %9, 4
  %32 = icmp sgt i32 %9, 3
  %33 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %27
  br label %35

34:                                               ; preds = %161
  ret void

35:                                               ; preds = %161, %12
  %36 = phi i32 [ 0, %12 ], [ %162, %161 ]
  %37 = add i32 %36, %4
  %38 = icmp ult i32 %37, %22
  br i1 %38, label %39, label %161

39:                                               ; preds = %35
  %40 = zext i32 %37 to i64
  %41 = mul i64 %25, %40
  %42 = getelementptr inbounds i8, i8 addrspace(1)* %33, i64 %41
  %43 = mul i64 %29, %40
  %44 = add i64 %43, %16
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %44
  %46 = bitcast i8 addrspace(1)* %45 to bfloat addrspace(1)*
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %44
  %48 = bitcast i8 addrspace(1)* %47 to bfloat addrspace(1)*
  %49 = getelementptr inbounds bfloat, bfloat addrspace(1)* %46, i64 %30
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !56
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !56
  %54 = fpext bfloat %53 to float
  br i1 %10, label %55, label %109

55:                                               ; preds = %39
  br i1 %32, label %56, label %101

56:                                               ; preds = %56, %55
  %57 = phi float [ %98, %56 ], [ 0.000000e+00, %55 ]
  %58 = phi i32 [ %99, %56 ], [ 0, %55 ]
  %59 = shl nuw nsw i32 %58, 1
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %60
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !78
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !78
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !67
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #15
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !67
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #15
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #12
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !67
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #15
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #12
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !67
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #15
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #12
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !79

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #12
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !67
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !67
  br label %161

109:                                              ; preds = %109, %39
  %110 = phi float [ %151, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i32 [ %152, %109 ], [ 0, %39 ]
  %112 = shl nuw nsw i32 %111, 1
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %113
  %115 = load i8, i8 addrspace(1)* %114, align 1, !tbaa !78
  %116 = zext i8 %115 to i32
  %117 = or i32 %112, 1
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %118
  %120 = load i8, i8 addrspace(1)* %119, align 1, !tbaa !78
  %121 = zext i8 %120 to i32
  %122 = shl nuw nsw i32 %121, 8
  %123 = shl nuw nsw i32 %111, 2
  %124 = zext i32 %123 to i64
  %125 = getelementptr inbounds float, float* %7, i64 %124
  %126 = load float, float* %125, align 4, !tbaa !67
  %127 = and i32 %116, 15
  %128 = tail call float @air.convert.f.f32.s.i32(i32 %127) #15
  %129 = or i32 %123, 1
  %130 = zext i32 %129 to i64
  %131 = getelementptr inbounds float, float* %7, i64 %130
  %132 = load float, float* %131, align 4, !tbaa !67
  %133 = and i32 %116, 240
  %134 = tail call float @air.convert.f.f32.s.i32(i32 %133) #15
  %135 = fmul float %132, %134
  %136 = tail call float @llvm.fmuladd.f32(float %126, float %128, float %135) #12
  %137 = or i32 %123, 2
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds float, float* %7, i64 %138
  %140 = load float, float* %139, align 4, !tbaa !67
  %141 = and i32 %122, 3840
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #15
  %143 = tail call float @llvm.fmuladd.f32(float %140, float %142, float %136) #12
  %144 = or i32 %123, 3
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds float, float* %7, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !67
  %148 = and i32 %122, 61440
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #15
  %150 = tail call float @llvm.fmuladd.f32(float %147, float %149, float %143) #12
  %151 = fadd float %110, %150
  %152 = add nuw nsw i32 %111, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %154, label %109, !llvm.loop !80

154:                                              ; preds = %109
  %155 = fmul float %54, %8
  %156 = tail call float @llvm.fmuladd.f32(float %51, float %151, float %155) #12
  %157 = zext i32 %36 to i64
  %158 = getelementptr inbounds float, float* %11, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !67
  %160 = fadd float %159, %156
  store float %160, float* %158, align 4, !tbaa !67
  br label %161

161:                                              ; preds = %154, %101, %35
  %162 = add nuw nsw i32 %36, 1
  %163 = icmp eq i32 %162, 4
  br i1 %163, label %34, label %35, !llvm.loop !89
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt4ELt128ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #6 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 7
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = lshr i32 %5, 1
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %29 = load i64, i64 addrspace(2)* %28, align 8
  %30 = zext i32 %20 to i64
  %31 = sdiv i32 %9, 4
  %32 = icmp sgt i32 %9, 3
  %33 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %27
  br label %35

34:                                               ; preds = %160
  ret void

35:                                               ; preds = %160, %12
  %36 = phi i32 [ 0, %12 ], [ %161, %160 ]
  %37 = add i32 %36, %4
  %38 = icmp ult i32 %37, %22
  br i1 %38, label %39, label %160

39:                                               ; preds = %35
  %40 = zext i32 %37 to i64
  %41 = mul i64 %25, %40
  %42 = getelementptr inbounds i8, i8 addrspace(1)* %33, i64 %41
  %43 = mul i64 %29, %40
  %44 = add i64 %43, %16
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %44
  %46 = bitcast i8 addrspace(1)* %45 to bfloat addrspace(1)*
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %44
  %48 = bitcast i8 addrspace(1)* %47 to bfloat addrspace(1)*
  %49 = getelementptr inbounds bfloat, bfloat addrspace(1)* %46, i64 %30
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !56
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !56
  %54 = fpext bfloat %53 to float
  br i1 %10, label %55, label %109

55:                                               ; preds = %39
  br i1 %32, label %56, label %101

56:                                               ; preds = %56, %55
  %57 = phi float [ %98, %56 ], [ 0.000000e+00, %55 ]
  %58 = phi i32 [ %99, %56 ], [ 0, %55 ]
  %59 = shl nuw nsw i32 %58, 1
  %60 = zext i32 %59 to i64
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %60
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !78
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !78
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !67
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #15
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !67
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #15
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #12
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !67
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #15
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #12
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !67
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #15
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #12
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !82

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #12
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !67
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !67
  br label %160

109:                                              ; preds = %109, %39
  %110 = phi float [ %152, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i1 [ false, %109 ], [ true, %39 ]
  %112 = phi i32 [ 1, %109 ], [ 0, %39 ]
  %113 = shl nuw nsw i32 %112, 1
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %114
  %116 = load i8, i8 addrspace(1)* %115, align 1, !tbaa !78
  %117 = zext i8 %116 to i32
  %118 = or i32 %113, 1
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %119
  %121 = load i8, i8 addrspace(1)* %120, align 1, !tbaa !78
  %122 = zext i8 %121 to i32
  %123 = shl nuw nsw i32 %122, 8
  %124 = shl nuw nsw i32 %112, 2
  %125 = zext i32 %124 to i64
  %126 = getelementptr inbounds float, float* %7, i64 %125
  %127 = load float, float* %126, align 4, !tbaa !67
  %128 = and i32 %117, 15
  %129 = tail call float @air.convert.f.f32.s.i32(i32 %128) #15
  %130 = or i32 %124, 1
  %131 = zext i32 %130 to i64
  %132 = getelementptr inbounds float, float* %7, i64 %131
  %133 = load float, float* %132, align 4, !tbaa !67
  %134 = and i32 %117, 240
  %135 = tail call float @air.convert.f.f32.s.i32(i32 %134) #15
  %136 = fmul float %133, %135
  %137 = tail call float @llvm.fmuladd.f32(float %127, float %129, float %136) #12
  %138 = or i32 %124, 2
  %139 = zext i32 %138 to i64
  %140 = getelementptr inbounds float, float* %7, i64 %139
  %141 = load float, float* %140, align 4, !tbaa !67
  %142 = and i32 %123, 3840
  %143 = tail call float @air.convert.f.f32.s.i32(i32 %142) #15
  %144 = tail call float @llvm.fmuladd.f32(float %141, float %143, float %137) #12
  %145 = or i32 %124, 3
  %146 = zext i32 %145 to i64
  %147 = getelementptr inbounds float, float* %7, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !67
  %149 = and i32 %123, 61440
  %150 = tail call float @air.convert.f.f32.s.i32(i32 %149) #15
  %151 = tail call float @llvm.fmuladd.f32(float %148, float %150, float %144) #12
  %152 = fadd float %110, %151
  br i1 %111, label %109, label %153, !llvm.loop !83

153:                                              ; preds = %109
  %154 = fmul float %54, %8
  %155 = tail call float @llvm.fmuladd.f32(float %51, float %152, float %154) #12
  %156 = zext i32 %36 to i64
  %157 = getelementptr inbounds float, float* %11, i64 %156
  %158 = load float, float* %157, align 4, !tbaa !67
  %159 = fadd float %158, %155
  store float %159, float* %157, align 4, !tbaa !67
  br label %160

160:                                              ; preds = %153, %101, %35
  %161 = add nuw nsw i32 %36, 1
  %162 = icmp eq i32 %161, 4
  br i1 %162, label %34, label %35, !llvm.loop !90
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %11
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !47
  br label %39

22:                                               ; preds = %11
  %23 = shl i32 %10, 4
  %24 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %27 = load i64, i64 addrspace(2)* %26, align 8, !tbaa !54
  %28 = mul i64 %27, %25
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %30 = load i64, i64 addrspace(2)* %29, align 8, !tbaa !77
  %31 = mul i64 %30, %25
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !47
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %31
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %36 = load i64, i64 addrspace(2)* %35, align 8
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %38 = load i64, i64 addrspace(2)* %37, align 8
  br label %47

39:                                               ; preds = %152, %19
  %40 = phi i32 [ %21, %19 ], [ %33, %152 ]
  %41 = icmp eq i32 %10, 0
  %42 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %43 = zext i32 %40 to i64
  %44 = mul i64 %43, %9
  %45 = zext i32 %7 to i64
  %46 = add i64 %44, %45
  br label %156

47:                                               ; preds = %152, %22
  %48 = phi i32 [ 0, %22 ], [ %153, %152 ]
  %49 = add i32 %48, %23
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %50
  br label %52

52:                                               ; preds = %52, %47
  %53 = phi i1 [ true, %47 ], [ false, %52 ]
  %54 = phi i32 [ 0, %47 ], [ 8, %52 ]
  %55 = phi float [ 0.000000e+00, %47 ], [ %102, %52 ]
  %56 = zext i32 %54 to i64
  %57 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %56
  %58 = load bfloat, bfloat addrspace(1)* %57, align 2, !tbaa !56
  %59 = fpext bfloat %58 to float
  %60 = or i32 %54, 1
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %61
  %63 = load bfloat, bfloat addrspace(1)* %62, align 2, !tbaa !56
  %64 = fpext bfloat %63 to float
  %65 = fadd float %59, %64
  %66 = or i32 %54, 2
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %67
  %69 = load bfloat, bfloat addrspace(1)* %68, align 2, !tbaa !56
  %70 = fpext bfloat %69 to float
  %71 = fadd float %65, %70
  %72 = or i32 %54, 3
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %73
  %75 = load bfloat, bfloat addrspace(1)* %74, align 2, !tbaa !56
  %76 = fpext bfloat %75 to float
  %77 = fadd float %71, %76
  %78 = or i32 %54, 4
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %79
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !56
  %82 = fpext bfloat %81 to float
  %83 = fadd float %77, %82
  %84 = or i32 %54, 5
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %85
  %87 = load bfloat, bfloat addrspace(1)* %86, align 2, !tbaa !56
  %88 = fpext bfloat %87 to float
  %89 = fadd float %83, %88
  %90 = or i32 %54, 6
  %91 = zext i32 %90 to i64
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %91
  %93 = load bfloat, bfloat addrspace(1)* %92, align 2, !tbaa !56
  %94 = fpext bfloat %93 to float
  %95 = fadd float %89, %94
  %96 = or i32 %54, 7
  %97 = zext i32 %96 to i64
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %97
  %99 = load bfloat, bfloat addrspace(1)* %98, align 2, !tbaa !56
  %100 = fpext bfloat %99 to float
  %101 = fadd float %95, %100
  %102 = fadd float %55, %101
  %103 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %56
  store float %59, float* %103, align 4, !tbaa !67
  %104 = fmul float %64, 3.125000e-02
  %105 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %61
  store float %104, float* %105, align 4, !tbaa !67
  %106 = fmul float %70, 2.500000e-01
  %107 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %67
  store float %106, float* %107, align 4, !tbaa !67
  %108 = fmul float %76, 7.812500e-03
  %109 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %73
  store float %108, float* %109, align 4, !tbaa !67
  %110 = fmul float %82, 6.250000e-02
  %111 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %79
  store float %110, float* %111, align 4, !tbaa !67
  %112 = fmul float %88, 5.000000e-01
  %113 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %85
  store float %112, float* %113, align 4, !tbaa !67
  %114 = fmul float %94, 1.562500e-02
  %115 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %91
  store float %114, float* %115, align 4, !tbaa !67
  %116 = fmul float %100, 1.250000e-01
  %117 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %97
  store float %116, float* %117, align 4, !tbaa !67
  br i1 %53, label %52, label %118, !llvm.loop !91

118:                                              ; preds = %52
  %119 = lshr i32 %49, 6
  %120 = mul nuw nsw i64 %50, 5
  %121 = lshr exact i64 %120, 3
  %122 = zext i32 %119 to i64
  %123 = getelementptr inbounds i8, i8 addrspace(1)* %34, i64 %121
  br label %124

124:                                              ; preds = %149, %118
  %125 = phi i32 [ 0, %118 ], [ %150, %149 ]
  %126 = add i32 %125, %7
  %127 = icmp ult i32 %126, %33
  br i1 %127, label %128, label %149

128:                                              ; preds = %124
  %129 = zext i32 %126 to i64
  %130 = mul i64 %36, %129
  %131 = getelementptr inbounds i8, i8 addrspace(1)* %123, i64 %130
  %132 = mul i64 %38, %129
  %133 = add i64 %132, %28
  %134 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %133
  %135 = bitcast i8 addrspace(1)* %134 to bfloat addrspace(1)*
  %136 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %133
  %137 = bitcast i8 addrspace(1)* %136 to bfloat addrspace(1)*
  %138 = getelementptr inbounds bfloat, bfloat addrspace(1)* %135, i64 %122
  %139 = load bfloat, bfloat addrspace(1)* %138, align 2, !tbaa !56
  %140 = fpext bfloat %139 to float
  %141 = getelementptr inbounds bfloat, bfloat addrspace(1)* %137, i64 %122
  %142 = load bfloat, bfloat addrspace(1)* %141, align 2, !tbaa !56
  %143 = fpext bfloat %142 to float
  %144 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v123mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %131, float* noundef nonnull %24, float noundef %140, float noundef %143, float noundef %102) #16
  %145 = zext i32 %125 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !67
  %148 = fadd float %144, %147
  store float %148, float* %146, align 4, !tbaa !67
  br label %149

149:                                              ; preds = %128, %124
  %150 = add nuw nsw i32 %125, 1
  %151 = icmp eq i32 %150, 4
  br i1 %151, label %152, label %124, !llvm.loop !92

152:                                              ; preds = %149
  %153 = add i32 %48, 512
  %154 = icmp ult i32 %153, %17
  br i1 %154, label %47, label %39, !llvm.loop !93

155:                                              ; preds = %180
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #12
  ret void

156:                                              ; preds = %180, %39
  %157 = phi i32 [ 0, %39 ], [ %181, %180 ]
  %158 = add i32 %157, %7
  %159 = icmp ult i32 %158, %40
  br i1 %159, label %160, label %180

160:                                              ; preds = %156
  %161 = zext i32 %157 to i64
  %162 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %161
  %163 = load float, float* %162, align 4, !tbaa !67
  %164 = call fast float @air.simd_sum.f32(float %163) #14
  br i1 %41, label %165, label %180

165:                                              ; preds = %160
  %166 = fptrunc float %164 to bfloat
  %167 = bitcast float %164 to i32
  %168 = and i32 %167, 2139095040
  %169 = icmp eq i32 %168, 2139095040
  br i1 %169, label %175, label %170

170:                                              ; preds = %165
  %171 = fpext bfloat %166 to float
  %172 = bitcast float %171 to i32
  %173 = and i32 %172, 2139095040
  %174 = icmp eq i32 %173, 2139095040
  br i1 %174, label %175, label %177

175:                                              ; preds = %170, %165
  %176 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %42, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %177

177:                                              ; preds = %175, %170
  %178 = add i64 %46, %161
  %179 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %178
  store bfloat %166, bfloat addrspace(1)* %179, align 2, !tbaa !56
  br label %180

180:                                              ; preds = %177, %160, %156
  %181 = add nuw nsw i32 %157, 1
  %182 = icmp eq i32 %181, 4
  br i1 %182, label %155, label %156, !llvm.loop !94
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 256
  %19 = shl i32 %10, 3
  br i1 %18, label %20, label %181

20:                                               ; preds = %11
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 2
  %23 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 4
  %24 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 6
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %27 = load i64, i64 addrspace(2)* %26, align 8, !tbaa !54
  %28 = mul i64 %27, %25
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %30 = load i64, i64 addrspace(2)* %29, align 8, !tbaa !77
  %31 = mul i64 %30, %25
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !47
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %31
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %36 = load i64, i64 addrspace(2)* %35, align 8
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %38 = load i64, i64 addrspace(2)* %37, align 8
  br label %39

39:                                               ; preds = %170, %20
  %40 = phi i32 [ 0, %20 ], [ %171, %170 ]
  %41 = add i32 %40, %19
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %42
  %44 = load bfloat, bfloat addrspace(1)* %43, align 2, !tbaa !56
  %45 = fpext bfloat %44 to float
  %46 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 1
  %47 = load bfloat, bfloat addrspace(1)* %46, align 2, !tbaa !56
  %48 = fpext bfloat %47 to float
  %49 = fadd float %45, %48
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 2
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !56
  %52 = fpext bfloat %51 to float
  %53 = fadd float %49, %52
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 3
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !56
  %56 = fpext bfloat %55 to float
  %57 = fadd float %53, %56
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 4
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !56
  %60 = fpext bfloat %59 to float
  %61 = fadd float %57, %60
  %62 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 5
  %63 = load bfloat, bfloat addrspace(1)* %62, align 2, !tbaa !56
  %64 = fpext bfloat %63 to float
  %65 = fadd float %61, %64
  %66 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 6
  %67 = load bfloat, bfloat addrspace(1)* %66, align 2, !tbaa !56
  %68 = fpext bfloat %67 to float
  %69 = fadd float %65, %68
  %70 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 7
  %71 = load bfloat, bfloat addrspace(1)* %70, align 2, !tbaa !56
  %72 = fpext bfloat %71 to float
  %73 = fadd float %69, %72
  %74 = fadd float %73, 0.000000e+00
  %75 = fmul float %48, 3.125000e-02
  %76 = fmul float %52, 2.500000e-01
  %77 = fmul float %56, 7.812500e-03
  %78 = fmul float %60, 6.250000e-02
  %79 = fmul float %64, 5.000000e-01
  %80 = fmul float %68, 1.562500e-02
  %81 = fmul float %72, 1.250000e-01
  %82 = lshr i32 %41, 6
  %83 = mul nuw nsw i64 %42, 5
  %84 = lshr exact i64 %83, 3
  %85 = zext i32 %82 to i64
  %86 = getelementptr inbounds i8, i8 addrspace(1)* %34, i64 %84
  %87 = fmul float %75, 2.560000e+02
  %88 = fmul float %77, 2.560000e+02
  %89 = fmul float %78, 2.560000e+02
  %90 = fmul float %80, 2.560000e+02
  br label %91

91:                                               ; preds = %167, %39
  %92 = phi i32 [ 0, %39 ], [ %168, %167 ]
  %93 = add i32 %92, %7
  %94 = icmp ult i32 %93, %33
  br i1 %94, label %95, label %167

95:                                               ; preds = %91
  %96 = zext i32 %93 to i64
  %97 = mul i64 %36, %96
  %98 = getelementptr inbounds i8, i8 addrspace(1)* %86, i64 %97
  %99 = mul i64 %38, %96
  %100 = add i64 %99, %28
  %101 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %100
  %102 = bitcast i8 addrspace(1)* %101 to bfloat addrspace(1)*
  %103 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %100
  %104 = bitcast i8 addrspace(1)* %103 to bfloat addrspace(1)*
  %105 = getelementptr inbounds bfloat, bfloat addrspace(1)* %102, i64 %85
  %106 = load bfloat, bfloat addrspace(1)* %105, align 2, !tbaa !56
  %107 = fpext bfloat %106 to float
  %108 = getelementptr inbounds bfloat, bfloat addrspace(1)* %104, i64 %85
  %109 = load bfloat, bfloat addrspace(1)* %108, align 2, !tbaa !56
  %110 = fpext bfloat %109 to float
  %111 = load i8, i8 addrspace(1)* %98, align 1, !tbaa !78
  %112 = zext i8 %111 to i32
  %113 = and i32 %112, 31
  %114 = tail call float @air.convert.f.f32.s.i32(i32 %113) #15
  %115 = and i32 %112, 224
  %116 = tail call float @air.convert.f.f32.s.i32(i32 %115) #15
  %117 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 1
  %118 = load i8, i8 addrspace(1)* %117, align 1, !tbaa !78
  %119 = zext i8 %118 to i32
  %120 = and i32 %119, 3
  %121 = tail call float @air.convert.f.f32.s.i32(i32 %120) #15
  %122 = and i32 %119, 124
  %123 = tail call float @air.convert.f.f32.s.i32(i32 %122) #15
  %124 = and i32 %119, 128
  %125 = tail call float @air.convert.f.f32.s.i32(i32 %124) #15
  %126 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 2
  %127 = load i8, i8 addrspace(1)* %126, align 1, !tbaa !78
  %128 = zext i8 %127 to i32
  %129 = and i32 %128, 15
  %130 = tail call float @air.convert.f.f32.s.i32(i32 %129) #15
  %131 = and i32 %128, 240
  %132 = tail call float @air.convert.f.f32.s.i32(i32 %131) #15
  %133 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 3
  %134 = load i8, i8 addrspace(1)* %133, align 1, !tbaa !78
  %135 = zext i8 %134 to i32
  %136 = and i32 %135, 1
  %137 = tail call float @air.convert.f.f32.s.i32(i32 %136) #15
  %138 = and i32 %135, 62
  %139 = tail call float @air.convert.f.f32.s.i32(i32 %138) #15
  %140 = and i32 %135, 192
  %141 = tail call float @air.convert.f.f32.s.i32(i32 %140) #15
  %142 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 4
  %143 = load i8, i8 addrspace(1)* %142, align 1, !tbaa !78
  %144 = zext i8 %143 to i32
  %145 = and i32 %144, 7
  %146 = tail call float @air.convert.f.f32.s.i32(i32 %145) #15
  %147 = and i32 %144, 248
  %148 = tail call float @air.convert.f.f32.s.i32(i32 %147) #15
  %149 = tail call float @llvm.fmuladd.f32(float %114, float %45, float 0.000000e+00) #12
  %150 = tail call float @llvm.fmuladd.f32(float %116, float %75, float %149) #12
  %151 = tail call float @llvm.fmuladd.f32(float %121, float %87, float %150) #12
  %152 = tail call float @llvm.fmuladd.f32(float %123, float %76, float %151) #12
  %153 = tail call float @llvm.fmuladd.f32(float %125, float %77, float %152) #12
  %154 = tail call float @llvm.fmuladd.f32(float %130, float %88, float %153) #12
  %155 = tail call float @llvm.fmuladd.f32(float %132, float %78, float %154) #12
  %156 = tail call float @llvm.fmuladd.f32(float %137, float %89, float %155) #12
  %157 = tail call float @llvm.fmuladd.f32(float %139, float %79, float %156) #12
  %158 = tail call float @llvm.fmuladd.f32(float %141, float %80, float %157) #12
  %159 = tail call float @llvm.fmuladd.f32(float %146, float %90, float %158) #12
  %160 = tail call float @llvm.fmuladd.f32(float %148, float %81, float %159) #12
  %161 = fmul float %74, %110
  %162 = tail call float @llvm.fmuladd.f32(float %107, float %160, float %161) #12
  %163 = zext i32 %92 to i64
  %164 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %163
  %165 = load float, float* %164, align 4, !tbaa !67
  %166 = fadd float %165, %162
  store float %166, float* %164, align 4, !tbaa !67
  br label %167

167:                                              ; preds = %95, %91
  %168 = add nuw nsw i32 %92, 1
  %169 = icmp eq i32 %168, 4
  br i1 %169, label %170, label %91, !llvm.loop !95

170:                                              ; preds = %167
  %171 = add i32 %40, 256
  %172 = icmp ugt i32 %17, %171
  %173 = sub i32 %17, %171
  %174 = icmp ugt i32 %173, 256
  %175 = and i1 %172, %174
  br i1 %175, label %39, label %176, !llvm.loop !96

176:                                              ; preds = %170
  %177 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 1
  %178 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 3
  %179 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 5
  %180 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 7
  store float %45, float* %21, align 4, !tbaa !67
  store float %75, float* %177, align 4, !tbaa !67
  store float %76, float* %22, align 4, !tbaa !67
  store float %77, float* %178, align 4, !tbaa !67
  store float %78, float* %23, align 4, !tbaa !67
  store float %79, float* %179, align 4, !tbaa !67
  store float %80, float* %24, align 4, !tbaa !67
  store float %81, float* %180, align 4, !tbaa !67
  br label %181

181:                                              ; preds = %176, %11
  %182 = phi i32 [ %171, %176 ], [ 0, %11 ]
  %183 = phi i32 [ %173, %176 ], [ %17, %11 ]
  %184 = icmp ugt i32 %183, %19
  br i1 %184, label %185, label %188

185:                                              ; preds = %181
  %186 = sub i32 %183, %19
  %187 = tail call i32 @air.min.u.i32(i32 %186, i32 8) #15
  br label %188

188:                                              ; preds = %185, %181
  %189 = phi i32 [ %187, %185 ], [ 0, %181 ]
  %190 = icmp eq i32 %189, 0
  br i1 %190, label %191, label %194

191:                                              ; preds = %188
  %192 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %193 = load i32, i32 addrspace(2)* %192, align 4, !tbaa !47
  br label %247

194:                                              ; preds = %188
  %195 = add i32 %182, %19
  %196 = zext i32 %195 to i64
  %197 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %196
  %198 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %199 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v135mlx_qmv_f32xsum_v1_load_vector_safeIDF16bfLi8ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_i(bfloat addrspace(1)* noundef %197, float* noundef nonnull %198, i32 noundef %189) #13
  %200 = zext i32 %8 to i64
  %201 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %202 = load i64, i64 addrspace(2)* %201, align 8, !tbaa !54
  %203 = mul i64 %202, %200
  %204 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %205 = load i64, i64 addrspace(2)* %204, align 8, !tbaa !77
  %206 = mul i64 %205, %200
  %207 = lshr i32 %195, 6
  %208 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %209 = load i32, i32 addrspace(2)* %208, align 4, !tbaa !47
  %210 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %206
  %211 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %212 = load i64, i64 addrspace(2)* %211, align 8
  %213 = mul nuw nsw i64 %196, 5
  %214 = lshr exact i64 %213, 3
  %215 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %216 = load i64, i64 addrspace(2)* %215, align 8
  %217 = zext i32 %207 to i64
  %218 = getelementptr inbounds i8, i8 addrspace(1)* %210, i64 %214
  br label %219

219:                                              ; preds = %244, %194
  %220 = phi i32 [ 0, %194 ], [ %245, %244 ]
  %221 = add i32 %220, %7
  %222 = icmp ult i32 %221, %209
  br i1 %222, label %223, label %244

223:                                              ; preds = %219
  %224 = zext i32 %221 to i64
  %225 = mul i64 %212, %224
  %226 = getelementptr inbounds i8, i8 addrspace(1)* %218, i64 %225
  %227 = mul i64 %216, %224
  %228 = add i64 %227, %203
  %229 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %228
  %230 = bitcast i8 addrspace(1)* %229 to bfloat addrspace(1)*
  %231 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %228
  %232 = bitcast i8 addrspace(1)* %231 to bfloat addrspace(1)*
  %233 = getelementptr inbounds bfloat, bfloat addrspace(1)* %230, i64 %217
  %234 = load bfloat, bfloat addrspace(1)* %233, align 2, !tbaa !56
  %235 = fpext bfloat %234 to float
  %236 = getelementptr inbounds bfloat, bfloat addrspace(1)* %232, i64 %217
  %237 = load bfloat, bfloat addrspace(1)* %236, align 2, !tbaa !56
  %238 = fpext bfloat %237 to float
  %239 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %226, float* noundef nonnull %198, float noundef %235, float noundef %238, float noundef %199, i32 noundef %189) #16
  %240 = zext i32 %220 to i64
  %241 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %240
  %242 = load float, float* %241, align 4, !tbaa !67
  %243 = fadd float %239, %242
  store float %243, float* %241, align 4, !tbaa !67
  br label %244

244:                                              ; preds = %223, %219
  %245 = add nuw nsw i32 %220, 1
  %246 = icmp eq i32 %245, 4
  br i1 %246, label %247, label %219, !llvm.loop !95

247:                                              ; preds = %244, %191
  %248 = phi i32 [ %193, %191 ], [ %209, %244 ]
  %249 = icmp eq i32 %10, 0
  %250 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %251 = zext i32 %248 to i64
  %252 = mul i64 %251, %9
  %253 = zext i32 %7 to i64
  %254 = add i64 %252, %253
  br label %256

255:                                              ; preds = %280
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

256:                                              ; preds = %280, %247
  %257 = phi i32 [ 0, %247 ], [ %281, %280 ]
  %258 = add i32 %257, %7
  %259 = icmp ult i32 %258, %248
  br i1 %259, label %260, label %280

260:                                              ; preds = %256
  %261 = zext i32 %257 to i64
  %262 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %261
  %263 = load float, float* %262, align 4, !tbaa !67
  %264 = call fast float @air.simd_sum.f32(float %263) #14
  br i1 %249, label %265, label %280

265:                                              ; preds = %260
  %266 = fptrunc float %264 to bfloat
  %267 = bitcast float %264 to i32
  %268 = and i32 %267, 2139095040
  %269 = icmp eq i32 %268, 2139095040
  br i1 %269, label %275, label %270

270:                                              ; preds = %265
  %271 = fpext bfloat %266 to float
  %272 = bitcast float %271 to i32
  %273 = and i32 %272, 2139095040
  %274 = icmp eq i32 %273, 2139095040
  br i1 %274, label %275, label %277

275:                                              ; preds = %270, %265
  %276 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %250, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %277

277:                                              ; preds = %275, %270
  %278 = add i64 %254, %261
  %279 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %278
  store bfloat %266, bfloat addrspace(1)* %279, align 2, !tbaa !56
  br label %280

280:                                              ; preds = %277, %260, %256
  %281 = add nuw nsw i32 %257, 1
  %282 = icmp eq i32 %281, 4
  br i1 %282, label %255, label %256, !llvm.loop !97
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN25splash_mlx_qmv_f32xsum_v123mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4) local_unnamed_addr #6 {
  br label %9

6:                                                ; preds = %9
  %7 = fmul float %3, %4
  %8 = tail call float @llvm.fmuladd.f32(float %2, float %89, float %7)
  ret float %8

9:                                                ; preds = %9, %5
  %10 = phi i1 [ true, %5 ], [ false, %9 ]
  %11 = phi i32 [ 0, %5 ], [ 1, %9 ]
  %12 = phi float [ 0.000000e+00, %5 ], [ %89, %9 ]
  %13 = phi i8 addrspace(1)* [ %0, %5 ], [ %20, %9 ]
  %14 = phi float* [ %1, %5 ], [ %17, %9 ]
  %15 = shl nuw nsw i32 %11, 3
  %16 = zext i32 %15 to i64
  %17 = getelementptr inbounds float, float* %14, i64 %16
  %18 = mul nuw nsw i32 %11, 5
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds i8, i8 addrspace(1)* %13, i64 %19
  %21 = load i8, i8 addrspace(1)* %20, align 1, !tbaa !78
  %22 = zext i8 %21 to i32
  %23 = and i32 %22, 31
  %24 = tail call float @air.convert.f.f32.s.i32(i32 %23) #15
  %25 = load float, float* %17, align 4, !tbaa !67
  %26 = tail call float @llvm.fmuladd.f32(float %24, float %25, float %12)
  %27 = and i32 %22, 224
  %28 = tail call float @air.convert.f.f32.s.i32(i32 %27) #15
  %29 = getelementptr inbounds float, float* %17, i64 1
  %30 = load float, float* %29, align 4, !tbaa !67
  %31 = tail call float @llvm.fmuladd.f32(float %28, float %30, float %26)
  %32 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 1
  %33 = load i8, i8 addrspace(1)* %32, align 1, !tbaa !78
  %34 = zext i8 %33 to i32
  %35 = and i32 %34, 3
  %36 = tail call float @air.convert.f.f32.s.i32(i32 %35) #15
  %37 = fmul float %30, 2.560000e+02
  %38 = tail call float @llvm.fmuladd.f32(float %36, float %37, float %31)
  %39 = and i32 %34, 124
  %40 = tail call float @air.convert.f.f32.s.i32(i32 %39) #15
  %41 = getelementptr inbounds float, float* %17, i64 2
  %42 = load float, float* %41, align 4, !tbaa !67
  %43 = tail call float @llvm.fmuladd.f32(float %40, float %42, float %38)
  %44 = and i32 %34, 128
  %45 = tail call float @air.convert.f.f32.s.i32(i32 %44) #15
  %46 = getelementptr inbounds float, float* %17, i64 3
  %47 = load float, float* %46, align 4, !tbaa !67
  %48 = tail call float @llvm.fmuladd.f32(float %45, float %47, float %43)
  %49 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 2
  %50 = load i8, i8 addrspace(1)* %49, align 1, !tbaa !78
  %51 = zext i8 %50 to i32
  %52 = and i32 %51, 15
  %53 = tail call float @air.convert.f.f32.s.i32(i32 %52) #15
  %54 = fmul float %47, 2.560000e+02
  %55 = tail call float @llvm.fmuladd.f32(float %53, float %54, float %48)
  %56 = and i32 %51, 240
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %56) #15
  %58 = getelementptr inbounds float, float* %17, i64 4
  %59 = load float, float* %58, align 4, !tbaa !67
  %60 = tail call float @llvm.fmuladd.f32(float %57, float %59, float %55)
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 3
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !78
  %63 = zext i8 %62 to i32
  %64 = and i32 %63, 1
  %65 = tail call float @air.convert.f.f32.s.i32(i32 %64) #15
  %66 = fmul float %59, 2.560000e+02
  %67 = tail call float @llvm.fmuladd.f32(float %65, float %66, float %60)
  %68 = and i32 %63, 62
  %69 = tail call float @air.convert.f.f32.s.i32(i32 %68) #15
  %70 = getelementptr inbounds float, float* %17, i64 5
  %71 = load float, float* %70, align 4, !tbaa !67
  %72 = tail call float @llvm.fmuladd.f32(float %69, float %71, float %67)
  %73 = and i32 %63, 192
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %73) #15
  %75 = getelementptr inbounds float, float* %17, i64 6
  %76 = load float, float* %75, align 4, !tbaa !67
  %77 = tail call float @llvm.fmuladd.f32(float %74, float %76, float %72)
  %78 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 4
  %79 = load i8, i8 addrspace(1)* %78, align 1, !tbaa !78
  %80 = zext i8 %79 to i32
  %81 = and i32 %80, 7
  %82 = tail call float @air.convert.f.f32.s.i32(i32 %81) #15
  %83 = fmul float %76, 2.560000e+02
  %84 = tail call float @llvm.fmuladd.f32(float %82, float %83, float %77)
  %85 = and i32 %80, 248
  %86 = tail call float @air.convert.f.f32.s.i32(i32 %85) #15
  %87 = getelementptr inbounds float, float* %17, i64 7
  %88 = load float, float* %87, align 4, !tbaa !67
  %89 = tail call float @llvm.fmuladd.f32(float %86, float %88, float %84)
  br i1 %10, label %9, label %6, !llvm.loop !98
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN25splash_mlx_qmv_f32xsum_v135mlx_qmv_f32xsum_v1_load_vector_safeIDF16bfLi8ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_i(bfloat addrspace(1)* noundef %0, float* noundef %1, i32 noundef %2) local_unnamed_addr #6 {
  %4 = icmp sgt i32 %2, 0
  br i1 %4, label %8, label %5

5:                                                ; preds = %8, %3
  %6 = phi float [ 0.000000e+00, %3 ], [ %57, %8 ]
  %7 = icmp slt i32 %2, 8
  br i1 %7, label %76, label %75

8:                                                ; preds = %8, %3
  %9 = phi i32 [ %73, %8 ], [ 0, %3 ]
  %10 = phi float [ %57, %8 ], [ 0.000000e+00, %3 ]
  %11 = zext i32 %9 to i64
  %12 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %11
  %13 = load bfloat, bfloat addrspace(1)* %12, align 2, !tbaa !56
  %14 = fpext bfloat %13 to float
  %15 = or i32 %9, 1
  %16 = zext i32 %15 to i64
  %17 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %16
  %18 = load bfloat, bfloat addrspace(1)* %17, align 2, !tbaa !56
  %19 = fpext bfloat %18 to float
  %20 = fadd float %14, %19
  %21 = or i32 %9, 2
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %22
  %24 = load bfloat, bfloat addrspace(1)* %23, align 2, !tbaa !56
  %25 = fpext bfloat %24 to float
  %26 = fadd float %20, %25
  %27 = or i32 %9, 3
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %28
  %30 = load bfloat, bfloat addrspace(1)* %29, align 2, !tbaa !56
  %31 = fpext bfloat %30 to float
  %32 = fadd float %26, %31
  %33 = or i32 %9, 4
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %34
  %36 = load bfloat, bfloat addrspace(1)* %35, align 2, !tbaa !56
  %37 = fpext bfloat %36 to float
  %38 = fadd float %32, %37
  %39 = or i32 %9, 5
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %40
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !56
  %43 = fpext bfloat %42 to float
  %44 = fadd float %38, %43
  %45 = or i32 %9, 6
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !56
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %9, 7
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !56
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = fadd float %10, %56
  %58 = getelementptr inbounds float, float* %1, i64 %11
  store float %14, float* %58, align 4, !tbaa !67
  %59 = fmul float %19, 3.125000e-02
  %60 = getelementptr inbounds float, float* %1, i64 %16
  store float %59, float* %60, align 4, !tbaa !67
  %61 = fmul float %25, 2.500000e-01
  %62 = getelementptr inbounds float, float* %1, i64 %22
  store float %61, float* %62, align 4, !tbaa !67
  %63 = fmul float %31, 7.812500e-03
  %64 = getelementptr inbounds float, float* %1, i64 %28
  store float %63, float* %64, align 4, !tbaa !67
  %65 = fmul float %37, 6.250000e-02
  %66 = getelementptr inbounds float, float* %1, i64 %34
  store float %65, float* %66, align 4, !tbaa !67
  %67 = fmul float %43, 5.000000e-01
  %68 = getelementptr inbounds float, float* %1, i64 %40
  store float %67, float* %68, align 4, !tbaa !67
  %69 = fmul float %49, 1.562500e-02
  %70 = getelementptr inbounds float, float* %1, i64 %46
  store float %69, float* %70, align 4, !tbaa !67
  %71 = fmul float %55, 1.250000e-01
  %72 = getelementptr inbounds float, float* %1, i64 %52
  store float %71, float* %72, align 4, !tbaa !67
  %73 = add nuw nsw i32 %9, 8
  %74 = icmp slt i32 %73, %2
  br i1 %74, label %8, label %5, !llvm.loop !99

75:                                               ; preds = %76, %5
  ret float %6

76:                                               ; preds = %76, %5
  %77 = phi i32 [ %80, %76 ], [ %2, %5 ]
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds float, float* %1, i64 %78
  store float 0.000000e+00, float* %79, align 4, !tbaa !67
  %80 = add i32 %77, 1
  %81 = icmp eq i32 %80, 8
  br i1 %81, label %75, label %76, !llvm.loop !100
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #6 {
  %7 = sdiv i32 %5, 8
  %8 = icmp sgt i32 %5, 7
  br i1 %8, label %13, label %9

9:                                                ; preds = %13, %6
  %10 = phi float [ 0.000000e+00, %6 ], [ %92, %13 ]
  %11 = fmul float %3, %4
  %12 = tail call float @llvm.fmuladd.f32(float %2, float %10, float %11)
  ret float %12

13:                                               ; preds = %13, %6
  %14 = phi i32 [ %93, %13 ], [ 0, %6 ]
  %15 = phi float [ %92, %13 ], [ 0.000000e+00, %6 ]
  %16 = phi i8 addrspace(1)* [ %23, %13 ], [ %0, %6 ]
  %17 = phi float* [ %20, %13 ], [ %1, %6 ]
  %18 = shl nsw i32 %14, 3
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds float, float* %17, i64 %19
  %21 = mul nuw nsw i32 %14, 5
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %16, i64 %22
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !78
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 31
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #15
  %28 = load float, float* %20, align 4, !tbaa !67
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 224
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #15
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !67
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !78
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 3
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #15
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 124
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #15
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !67
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = and i32 %37, 128
  %48 = tail call float @air.convert.f.f32.s.i32(i32 %47) #15
  %49 = getelementptr inbounds float, float* %20, i64 3
  %50 = load float, float* %49, align 4, !tbaa !67
  %51 = tail call float @llvm.fmuladd.f32(float %48, float %50, float %46)
  %52 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %53 = load i8, i8 addrspace(1)* %52, align 1, !tbaa !78
  %54 = zext i8 %53 to i32
  %55 = and i32 %54, 15
  %56 = tail call float @air.convert.f.f32.s.i32(i32 %55) #15
  %57 = fmul float %50, 2.560000e+02
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %51)
  %59 = and i32 %54, 240
  %60 = tail call float @air.convert.f.f32.s.i32(i32 %59) #15
  %61 = getelementptr inbounds float, float* %20, i64 4
  %62 = load float, float* %61, align 4, !tbaa !67
  %63 = tail call float @llvm.fmuladd.f32(float %60, float %62, float %58)
  %64 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 3
  %65 = load i8, i8 addrspace(1)* %64, align 1, !tbaa !78
  %66 = zext i8 %65 to i32
  %67 = and i32 %66, 1
  %68 = tail call float @air.convert.f.f32.s.i32(i32 %67) #15
  %69 = fmul float %62, 2.560000e+02
  %70 = tail call float @llvm.fmuladd.f32(float %68, float %69, float %63)
  %71 = and i32 %66, 62
  %72 = tail call float @air.convert.f.f32.s.i32(i32 %71) #15
  %73 = getelementptr inbounds float, float* %20, i64 5
  %74 = load float, float* %73, align 4, !tbaa !67
  %75 = tail call float @llvm.fmuladd.f32(float %72, float %74, float %70)
  %76 = and i32 %66, 192
  %77 = tail call float @air.convert.f.f32.s.i32(i32 %76) #15
  %78 = getelementptr inbounds float, float* %20, i64 6
  %79 = load float, float* %78, align 4, !tbaa !67
  %80 = tail call float @llvm.fmuladd.f32(float %77, float %79, float %75)
  %81 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 4
  %82 = load i8, i8 addrspace(1)* %81, align 1, !tbaa !78
  %83 = zext i8 %82 to i32
  %84 = and i32 %83, 7
  %85 = tail call float @air.convert.f.f32.s.i32(i32 %84) #15
  %86 = fmul float %79, 2.560000e+02
  %87 = tail call float @llvm.fmuladd.f32(float %85, float %86, float %80)
  %88 = and i32 %83, 248
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #15
  %90 = getelementptr inbounds float, float* %20, i64 7
  %91 = load float, float* %90, align 4, !tbaa !67
  %92 = tail call float @llvm.fmuladd.f32(float %89, float %91, float %87)
  %93 = add nuw nsw i32 %14, 1
  %94 = icmp eq i32 %93, %7
  br i1 %94, label %9, label %13, !llvm.loop !101
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %11
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !47
  br label %39

22:                                               ; preds = %11
  %23 = shl i32 %10, 4
  %24 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %27 = load i64, i64 addrspace(2)* %26, align 8, !tbaa !54
  %28 = mul i64 %27, %25
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %30 = load i64, i64 addrspace(2)* %29, align 8, !tbaa !77
  %31 = mul i64 %30, %25
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !47
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %31
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %36 = load i64, i64 addrspace(2)* %35, align 8
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %38 = load i64, i64 addrspace(2)* %37, align 8
  br label %47

39:                                               ; preds = %152, %19
  %40 = phi i32 [ %21, %19 ], [ %33, %152 ]
  %41 = icmp eq i32 %10, 0
  %42 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %43 = zext i32 %40 to i64
  %44 = mul i64 %43, %9
  %45 = zext i32 %7 to i64
  %46 = add i64 %44, %45
  br label %156

47:                                               ; preds = %152, %22
  %48 = phi i32 [ 0, %22 ], [ %153, %152 ]
  %49 = add i32 %48, %23
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %50
  br label %52

52:                                               ; preds = %52, %47
  %53 = phi i1 [ true, %47 ], [ false, %52 ]
  %54 = phi i32 [ 0, %47 ], [ 8, %52 ]
  %55 = phi float [ 0.000000e+00, %47 ], [ %102, %52 ]
  %56 = zext i32 %54 to i64
  %57 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %56
  %58 = load bfloat, bfloat addrspace(1)* %57, align 2, !tbaa !56
  %59 = fpext bfloat %58 to float
  %60 = or i32 %54, 1
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %61
  %63 = load bfloat, bfloat addrspace(1)* %62, align 2, !tbaa !56
  %64 = fpext bfloat %63 to float
  %65 = fadd float %59, %64
  %66 = or i32 %54, 2
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %67
  %69 = load bfloat, bfloat addrspace(1)* %68, align 2, !tbaa !56
  %70 = fpext bfloat %69 to float
  %71 = fadd float %65, %70
  %72 = or i32 %54, 3
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %73
  %75 = load bfloat, bfloat addrspace(1)* %74, align 2, !tbaa !56
  %76 = fpext bfloat %75 to float
  %77 = fadd float %71, %76
  %78 = or i32 %54, 4
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %79
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !56
  %82 = fpext bfloat %81 to float
  %83 = fadd float %77, %82
  %84 = or i32 %54, 5
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %85
  %87 = load bfloat, bfloat addrspace(1)* %86, align 2, !tbaa !56
  %88 = fpext bfloat %87 to float
  %89 = fadd float %83, %88
  %90 = or i32 %54, 6
  %91 = zext i32 %90 to i64
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %91
  %93 = load bfloat, bfloat addrspace(1)* %92, align 2, !tbaa !56
  %94 = fpext bfloat %93 to float
  %95 = fadd float %89, %94
  %96 = or i32 %54, 7
  %97 = zext i32 %96 to i64
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %97
  %99 = load bfloat, bfloat addrspace(1)* %98, align 2, !tbaa !56
  %100 = fpext bfloat %99 to float
  %101 = fadd float %95, %100
  %102 = fadd float %55, %101
  %103 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %56
  store float %59, float* %103, align 4, !tbaa !67
  %104 = fmul float %64, 3.125000e-02
  %105 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %61
  store float %104, float* %105, align 4, !tbaa !67
  %106 = fmul float %70, 2.500000e-01
  %107 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %67
  store float %106, float* %107, align 4, !tbaa !67
  %108 = fmul float %76, 7.812500e-03
  %109 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %73
  store float %108, float* %109, align 4, !tbaa !67
  %110 = fmul float %82, 6.250000e-02
  %111 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %79
  store float %110, float* %111, align 4, !tbaa !67
  %112 = fmul float %88, 5.000000e-01
  %113 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %85
  store float %112, float* %113, align 4, !tbaa !67
  %114 = fmul float %94, 1.562500e-02
  %115 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %91
  store float %114, float* %115, align 4, !tbaa !67
  %116 = fmul float %100, 1.250000e-01
  %117 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %97
  store float %116, float* %117, align 4, !tbaa !67
  br i1 %53, label %52, label %118, !llvm.loop !91

118:                                              ; preds = %52
  %119 = lshr i32 %49, 7
  %120 = mul nuw nsw i64 %50, 5
  %121 = lshr exact i64 %120, 3
  %122 = zext i32 %119 to i64
  %123 = getelementptr inbounds i8, i8 addrspace(1)* %34, i64 %121
  br label %124

124:                                              ; preds = %149, %118
  %125 = phi i32 [ 0, %118 ], [ %150, %149 ]
  %126 = add i32 %125, %7
  %127 = icmp ult i32 %126, %33
  br i1 %127, label %128, label %149

128:                                              ; preds = %124
  %129 = zext i32 %126 to i64
  %130 = mul i64 %36, %129
  %131 = getelementptr inbounds i8, i8 addrspace(1)* %123, i64 %130
  %132 = mul i64 %38, %129
  %133 = add i64 %132, %28
  %134 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %133
  %135 = bitcast i8 addrspace(1)* %134 to bfloat addrspace(1)*
  %136 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %133
  %137 = bitcast i8 addrspace(1)* %136 to bfloat addrspace(1)*
  %138 = getelementptr inbounds bfloat, bfloat addrspace(1)* %135, i64 %122
  %139 = load bfloat, bfloat addrspace(1)* %138, align 2, !tbaa !56
  %140 = fpext bfloat %139 to float
  %141 = getelementptr inbounds bfloat, bfloat addrspace(1)* %137, i64 %122
  %142 = load bfloat, bfloat addrspace(1)* %141, align 2, !tbaa !56
  %143 = fpext bfloat %142 to float
  %144 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v123mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %131, float* noundef nonnull %24, float noundef %140, float noundef %143, float noundef %102) #16
  %145 = zext i32 %125 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !67
  %148 = fadd float %144, %147
  store float %148, float* %146, align 4, !tbaa !67
  br label %149

149:                                              ; preds = %128, %124
  %150 = add nuw nsw i32 %125, 1
  %151 = icmp eq i32 %150, 4
  br i1 %151, label %152, label %124, !llvm.loop !102

152:                                              ; preds = %149
  %153 = add i32 %48, 512
  %154 = icmp ult i32 %153, %17
  br i1 %154, label %47, label %39, !llvm.loop !103

155:                                              ; preds = %180
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #12
  ret void

156:                                              ; preds = %180, %39
  %157 = phi i32 [ 0, %39 ], [ %181, %180 ]
  %158 = add i32 %157, %7
  %159 = icmp ult i32 %158, %40
  br i1 %159, label %160, label %180

160:                                              ; preds = %156
  %161 = zext i32 %157 to i64
  %162 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %161
  %163 = load float, float* %162, align 4, !tbaa !67
  %164 = call fast float @air.simd_sum.f32(float %163) #14
  br i1 %41, label %165, label %180

165:                                              ; preds = %160
  %166 = fptrunc float %164 to bfloat
  %167 = bitcast float %164 to i32
  %168 = and i32 %167, 2139095040
  %169 = icmp eq i32 %168, 2139095040
  br i1 %169, label %175, label %170

170:                                              ; preds = %165
  %171 = fpext bfloat %166 to float
  %172 = bitcast float %171 to i32
  %173 = and i32 %172, 2139095040
  %174 = icmp eq i32 %173, 2139095040
  br i1 %174, label %175, label %177

175:                                              ; preds = %170, %165
  %176 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %42, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %177

177:                                              ; preds = %175, %170
  %178 = add i64 %46, %161
  %179 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %178
  store bfloat %166, bfloat addrspace(1)* %179, align 2, !tbaa !56
  br label %180

180:                                              ; preds = %177, %160, %156
  %181 = add nuw nsw i32 %157, 1
  %182 = icmp eq i32 %181, 4
  br i1 %182, label %155, label %156, !llvm.loop !104
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt5ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 256
  %19 = shl i32 %10, 3
  br i1 %18, label %20, label %181

20:                                               ; preds = %11
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 2
  %23 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 4
  %24 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 6
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %27 = load i64, i64 addrspace(2)* %26, align 8, !tbaa !54
  %28 = mul i64 %27, %25
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %30 = load i64, i64 addrspace(2)* %29, align 8, !tbaa !77
  %31 = mul i64 %30, %25
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !47
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %31
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %36 = load i64, i64 addrspace(2)* %35, align 8
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %38 = load i64, i64 addrspace(2)* %37, align 8
  br label %39

39:                                               ; preds = %170, %20
  %40 = phi i32 [ 0, %20 ], [ %171, %170 ]
  %41 = add i32 %40, %19
  %42 = zext i32 %41 to i64
  %43 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %42
  %44 = load bfloat, bfloat addrspace(1)* %43, align 2, !tbaa !56
  %45 = fpext bfloat %44 to float
  %46 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 1
  %47 = load bfloat, bfloat addrspace(1)* %46, align 2, !tbaa !56
  %48 = fpext bfloat %47 to float
  %49 = fadd float %45, %48
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 2
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !56
  %52 = fpext bfloat %51 to float
  %53 = fadd float %49, %52
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 3
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !56
  %56 = fpext bfloat %55 to float
  %57 = fadd float %53, %56
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 4
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !56
  %60 = fpext bfloat %59 to float
  %61 = fadd float %57, %60
  %62 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 5
  %63 = load bfloat, bfloat addrspace(1)* %62, align 2, !tbaa !56
  %64 = fpext bfloat %63 to float
  %65 = fadd float %61, %64
  %66 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 6
  %67 = load bfloat, bfloat addrspace(1)* %66, align 2, !tbaa !56
  %68 = fpext bfloat %67 to float
  %69 = fadd float %65, %68
  %70 = getelementptr inbounds bfloat, bfloat addrspace(1)* %43, i64 7
  %71 = load bfloat, bfloat addrspace(1)* %70, align 2, !tbaa !56
  %72 = fpext bfloat %71 to float
  %73 = fadd float %69, %72
  %74 = fadd float %73, 0.000000e+00
  %75 = fmul float %48, 3.125000e-02
  %76 = fmul float %52, 2.500000e-01
  %77 = fmul float %56, 7.812500e-03
  %78 = fmul float %60, 6.250000e-02
  %79 = fmul float %64, 5.000000e-01
  %80 = fmul float %68, 1.562500e-02
  %81 = fmul float %72, 1.250000e-01
  %82 = lshr i32 %41, 7
  %83 = mul nuw nsw i64 %42, 5
  %84 = lshr exact i64 %83, 3
  %85 = zext i32 %82 to i64
  %86 = getelementptr inbounds i8, i8 addrspace(1)* %34, i64 %84
  %87 = fmul float %75, 2.560000e+02
  %88 = fmul float %77, 2.560000e+02
  %89 = fmul float %78, 2.560000e+02
  %90 = fmul float %80, 2.560000e+02
  br label %91

91:                                               ; preds = %167, %39
  %92 = phi i32 [ 0, %39 ], [ %168, %167 ]
  %93 = add i32 %92, %7
  %94 = icmp ult i32 %93, %33
  br i1 %94, label %95, label %167

95:                                               ; preds = %91
  %96 = zext i32 %93 to i64
  %97 = mul i64 %36, %96
  %98 = getelementptr inbounds i8, i8 addrspace(1)* %86, i64 %97
  %99 = mul i64 %38, %96
  %100 = add i64 %99, %28
  %101 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %100
  %102 = bitcast i8 addrspace(1)* %101 to bfloat addrspace(1)*
  %103 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %100
  %104 = bitcast i8 addrspace(1)* %103 to bfloat addrspace(1)*
  %105 = getelementptr inbounds bfloat, bfloat addrspace(1)* %102, i64 %85
  %106 = load bfloat, bfloat addrspace(1)* %105, align 2, !tbaa !56
  %107 = fpext bfloat %106 to float
  %108 = getelementptr inbounds bfloat, bfloat addrspace(1)* %104, i64 %85
  %109 = load bfloat, bfloat addrspace(1)* %108, align 2, !tbaa !56
  %110 = fpext bfloat %109 to float
  %111 = load i8, i8 addrspace(1)* %98, align 1, !tbaa !78
  %112 = zext i8 %111 to i32
  %113 = and i32 %112, 31
  %114 = tail call float @air.convert.f.f32.s.i32(i32 %113) #15
  %115 = and i32 %112, 224
  %116 = tail call float @air.convert.f.f32.s.i32(i32 %115) #15
  %117 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 1
  %118 = load i8, i8 addrspace(1)* %117, align 1, !tbaa !78
  %119 = zext i8 %118 to i32
  %120 = and i32 %119, 3
  %121 = tail call float @air.convert.f.f32.s.i32(i32 %120) #15
  %122 = and i32 %119, 124
  %123 = tail call float @air.convert.f.f32.s.i32(i32 %122) #15
  %124 = and i32 %119, 128
  %125 = tail call float @air.convert.f.f32.s.i32(i32 %124) #15
  %126 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 2
  %127 = load i8, i8 addrspace(1)* %126, align 1, !tbaa !78
  %128 = zext i8 %127 to i32
  %129 = and i32 %128, 15
  %130 = tail call float @air.convert.f.f32.s.i32(i32 %129) #15
  %131 = and i32 %128, 240
  %132 = tail call float @air.convert.f.f32.s.i32(i32 %131) #15
  %133 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 3
  %134 = load i8, i8 addrspace(1)* %133, align 1, !tbaa !78
  %135 = zext i8 %134 to i32
  %136 = and i32 %135, 1
  %137 = tail call float @air.convert.f.f32.s.i32(i32 %136) #15
  %138 = and i32 %135, 62
  %139 = tail call float @air.convert.f.f32.s.i32(i32 %138) #15
  %140 = and i32 %135, 192
  %141 = tail call float @air.convert.f.f32.s.i32(i32 %140) #15
  %142 = getelementptr inbounds i8, i8 addrspace(1)* %98, i64 4
  %143 = load i8, i8 addrspace(1)* %142, align 1, !tbaa !78
  %144 = zext i8 %143 to i32
  %145 = and i32 %144, 7
  %146 = tail call float @air.convert.f.f32.s.i32(i32 %145) #15
  %147 = and i32 %144, 248
  %148 = tail call float @air.convert.f.f32.s.i32(i32 %147) #15
  %149 = tail call float @llvm.fmuladd.f32(float %114, float %45, float 0.000000e+00) #12
  %150 = tail call float @llvm.fmuladd.f32(float %116, float %75, float %149) #12
  %151 = tail call float @llvm.fmuladd.f32(float %121, float %87, float %150) #12
  %152 = tail call float @llvm.fmuladd.f32(float %123, float %76, float %151) #12
  %153 = tail call float @llvm.fmuladd.f32(float %125, float %77, float %152) #12
  %154 = tail call float @llvm.fmuladd.f32(float %130, float %88, float %153) #12
  %155 = tail call float @llvm.fmuladd.f32(float %132, float %78, float %154) #12
  %156 = tail call float @llvm.fmuladd.f32(float %137, float %89, float %155) #12
  %157 = tail call float @llvm.fmuladd.f32(float %139, float %79, float %156) #12
  %158 = tail call float @llvm.fmuladd.f32(float %141, float %80, float %157) #12
  %159 = tail call float @llvm.fmuladd.f32(float %146, float %90, float %158) #12
  %160 = tail call float @llvm.fmuladd.f32(float %148, float %81, float %159) #12
  %161 = fmul float %74, %110
  %162 = tail call float @llvm.fmuladd.f32(float %107, float %160, float %161) #12
  %163 = zext i32 %92 to i64
  %164 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %163
  %165 = load float, float* %164, align 4, !tbaa !67
  %166 = fadd float %165, %162
  store float %166, float* %164, align 4, !tbaa !67
  br label %167

167:                                              ; preds = %95, %91
  %168 = add nuw nsw i32 %92, 1
  %169 = icmp eq i32 %168, 4
  br i1 %169, label %170, label %91, !llvm.loop !105

170:                                              ; preds = %167
  %171 = add i32 %40, 256
  %172 = icmp ugt i32 %17, %171
  %173 = sub i32 %17, %171
  %174 = icmp ugt i32 %173, 256
  %175 = and i1 %172, %174
  br i1 %175, label %39, label %176, !llvm.loop !106

176:                                              ; preds = %170
  %177 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 1
  %178 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 3
  %179 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 5
  %180 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 7
  store float %45, float* %21, align 4, !tbaa !67
  store float %75, float* %177, align 4, !tbaa !67
  store float %76, float* %22, align 4, !tbaa !67
  store float %77, float* %178, align 4, !tbaa !67
  store float %78, float* %23, align 4, !tbaa !67
  store float %79, float* %179, align 4, !tbaa !67
  store float %80, float* %24, align 4, !tbaa !67
  store float %81, float* %180, align 4, !tbaa !67
  br label %181

181:                                              ; preds = %176, %11
  %182 = phi i32 [ %171, %176 ], [ 0, %11 ]
  %183 = phi i32 [ %173, %176 ], [ %17, %11 ]
  %184 = icmp ugt i32 %183, %19
  br i1 %184, label %185, label %188

185:                                              ; preds = %181
  %186 = sub i32 %183, %19
  %187 = tail call i32 @air.min.u.i32(i32 %186, i32 8) #15
  br label %188

188:                                              ; preds = %185, %181
  %189 = phi i32 [ %187, %185 ], [ 0, %181 ]
  %190 = icmp eq i32 %189, 0
  br i1 %190, label %191, label %194

191:                                              ; preds = %188
  %192 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %193 = load i32, i32 addrspace(2)* %192, align 4, !tbaa !47
  br label %247

194:                                              ; preds = %188
  %195 = add i32 %182, %19
  %196 = zext i32 %195 to i64
  %197 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %196
  %198 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %199 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v135mlx_qmv_f32xsum_v1_load_vector_safeIDF16bfLi8ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_i(bfloat addrspace(1)* noundef %197, float* noundef nonnull %198, i32 noundef %189) #13
  %200 = zext i32 %8 to i64
  %201 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %202 = load i64, i64 addrspace(2)* %201, align 8, !tbaa !54
  %203 = mul i64 %202, %200
  %204 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %205 = load i64, i64 addrspace(2)* %204, align 8, !tbaa !77
  %206 = mul i64 %205, %200
  %207 = lshr i32 %195, 7
  %208 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %209 = load i32, i32 addrspace(2)* %208, align 4, !tbaa !47
  %210 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %206
  %211 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %212 = load i64, i64 addrspace(2)* %211, align 8
  %213 = mul nuw nsw i64 %196, 5
  %214 = lshr exact i64 %213, 3
  %215 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %216 = load i64, i64 addrspace(2)* %215, align 8
  %217 = zext i32 %207 to i64
  %218 = getelementptr inbounds i8, i8 addrspace(1)* %210, i64 %214
  br label %219

219:                                              ; preds = %244, %194
  %220 = phi i32 [ 0, %194 ], [ %245, %244 ]
  %221 = add i32 %220, %7
  %222 = icmp ult i32 %221, %209
  br i1 %222, label %223, label %244

223:                                              ; preds = %219
  %224 = zext i32 %221 to i64
  %225 = mul i64 %212, %224
  %226 = getelementptr inbounds i8, i8 addrspace(1)* %218, i64 %225
  %227 = mul i64 %216, %224
  %228 = add i64 %227, %203
  %229 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %228
  %230 = bitcast i8 addrspace(1)* %229 to bfloat addrspace(1)*
  %231 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %228
  %232 = bitcast i8 addrspace(1)* %231 to bfloat addrspace(1)*
  %233 = getelementptr inbounds bfloat, bfloat addrspace(1)* %230, i64 %217
  %234 = load bfloat, bfloat addrspace(1)* %233, align 2, !tbaa !56
  %235 = fpext bfloat %234 to float
  %236 = getelementptr inbounds bfloat, bfloat addrspace(1)* %232, i64 %217
  %237 = load bfloat, bfloat addrspace(1)* %236, align 2, !tbaa !56
  %238 = fpext bfloat %237 to float
  %239 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %226, float* noundef nonnull %198, float noundef %235, float noundef %238, float noundef %199, i32 noundef %189) #16
  %240 = zext i32 %220 to i64
  %241 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %240
  %242 = load float, float* %241, align 4, !tbaa !67
  %243 = fadd float %239, %242
  store float %243, float* %241, align 4, !tbaa !67
  br label %244

244:                                              ; preds = %223, %219
  %245 = add nuw nsw i32 %220, 1
  %246 = icmp eq i32 %245, 4
  br i1 %246, label %247, label %219, !llvm.loop !105

247:                                              ; preds = %244, %191
  %248 = phi i32 [ %193, %191 ], [ %209, %244 ]
  %249 = icmp eq i32 %10, 0
  %250 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %251 = zext i32 %248 to i64
  %252 = mul i64 %251, %9
  %253 = zext i32 %7 to i64
  %254 = add i64 %252, %253
  br label %256

255:                                              ; preds = %280
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

256:                                              ; preds = %280, %247
  %257 = phi i32 [ 0, %247 ], [ %281, %280 ]
  %258 = add i32 %257, %7
  %259 = icmp ult i32 %258, %248
  br i1 %259, label %260, label %280

260:                                              ; preds = %256
  %261 = zext i32 %257 to i64
  %262 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %261
  %263 = load float, float* %262, align 4, !tbaa !67
  %264 = call fast float @air.simd_sum.f32(float %263) #14
  br i1 %249, label %265, label %280

265:                                              ; preds = %260
  %266 = fptrunc float %264 to bfloat
  %267 = bitcast float %264 to i32
  %268 = and i32 %267, 2139095040
  %269 = icmp eq i32 %268, 2139095040
  br i1 %269, label %275, label %270

270:                                              ; preds = %265
  %271 = fpext bfloat %266 to float
  %272 = bitcast float %271 to i32
  %273 = and i32 %272, 2139095040
  %274 = icmp eq i32 %273, 2139095040
  br i1 %274, label %275, label %277

275:                                              ; preds = %270, %265
  %276 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %250, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %277

277:                                              ; preds = %275, %270
  %278 = add i64 %254, %261
  %279 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %278
  store bfloat %266, bfloat addrspace(1)* %279, align 2, !tbaa !56
  br label %280

280:                                              ; preds = %277, %260, %256
  %281 = add nuw nsw i32 %257, 1
  %282 = icmp eq i32 %281, 4
  br i1 %282, label %255, label %256, !llvm.loop !107
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %23, label %19

19:                                               ; preds = %11
  %20 = shl i32 %10, 3
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %32

23:                                               ; preds = %71, %11
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %10, 0
  %27 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %28 = zext i32 %25 to i64
  %29 = mul i64 %28, %9
  %30 = zext i32 %7 to i64
  %31 = add i64 %29, %30
  br label %75

32:                                               ; preds = %71, %19
  %33 = phi i32 [ 0, %19 ], [ %72, %71 ]
  %34 = add i32 %33, %20
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %35
  br label %37

37:                                               ; preds = %37, %32
  %38 = phi i1 [ true, %32 ], [ false, %37 ]
  %39 = phi i32 [ 0, %32 ], [ 4, %37 ]
  %40 = phi float [ 0.000000e+00, %32 ], [ %63, %37 ]
  %41 = zext i32 %39 to i64
  %42 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %41
  %43 = load bfloat, bfloat addrspace(1)* %42, align 2, !tbaa !56
  %44 = fpext bfloat %43 to float
  %45 = or i32 %39, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !56
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %39, 2
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !56
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = or i32 %39, 3
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %58
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !56
  %61 = fpext bfloat %60 to float
  %62 = fadd float %56, %61
  %63 = fadd float %40, %62
  %64 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %41
  store float %44, float* %64, align 4, !tbaa !67
  %65 = fmul float %49, 1.562500e-02
  %66 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %46
  store float %65, float* %66, align 4, !tbaa !67
  %67 = fmul float %55, 6.250000e-02
  %68 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %52
  store float %67, float* %68, align 4, !tbaa !67
  %69 = fmul float %61, 2.500000e-01
  %70 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %58
  store float %69, float* %70, align 4, !tbaa !67
  br i1 %38, label %37, label %71, !llvm.loop !108

71:                                               ; preds = %37
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %34, i32 noundef %8, float* noundef nonnull %21, float noundef %63, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %22) #10
  %72 = add i32 %33, 256
  %73 = icmp ult i32 %72, %17
  br i1 %73, label %32, label %23, !llvm.loop !109

74:                                               ; preds = %99
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

75:                                               ; preds = %99, %23
  %76 = phi i32 [ 0, %23 ], [ %100, %99 ]
  %77 = add i32 %76, %7
  %78 = icmp ult i32 %77, %25
  br i1 %78, label %79, label %99

79:                                               ; preds = %75
  %80 = zext i32 %76 to i64
  %81 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %80
  %82 = load float, float* %81, align 4, !tbaa !67
  %83 = call fast float @air.simd_sum.f32(float %82) #14
  br i1 %26, label %84, label %99

84:                                               ; preds = %79
  %85 = fptrunc float %83 to bfloat
  %86 = bitcast float %83 to i32
  %87 = and i32 %86, 2139095040
  %88 = icmp eq i32 %87, 2139095040
  br i1 %88, label %94, label %89

89:                                               ; preds = %84
  %90 = fpext bfloat %85 to float
  %91 = bitcast float %90 to i32
  %92 = and i32 %91, 2139095040
  %93 = icmp eq i32 %92, 2139095040
  br i1 %93, label %94, label %96

94:                                               ; preds = %89, %84
  %95 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %27, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %96

96:                                               ; preds = %94, %89
  %97 = add i64 %31, %80
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %97
  store bfloat %85, bfloat addrspace(1)* %98, align 2, !tbaa !56
  br label %99

99:                                               ; preds = %96, %79, %75
  %100 = add nuw nsw i32 %76, 1
  %101 = icmp eq i32 %100, 4
  br i1 %101, label %74, label %75, !llvm.loop !110
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [4 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [4 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 128
  %19 = shl i32 %10, 2
  br i1 %18, label %20, label %54

20:                                               ; preds = %11
  %21 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 1
  %23 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 2
  %24 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 3
  %25 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %26

26:                                               ; preds = %26, %20
  %27 = phi i32 [ 0, %20 ], [ %49, %26 ]
  %28 = add i32 %27, %19
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %29
  %31 = load bfloat, bfloat addrspace(1)* %30, align 2, !tbaa !56
  %32 = fpext bfloat %31 to float
  %33 = getelementptr inbounds bfloat, bfloat addrspace(1)* %30, i64 1
  %34 = load bfloat, bfloat addrspace(1)* %33, align 2, !tbaa !56
  %35 = fpext bfloat %34 to float
  %36 = fadd float %32, %35
  %37 = getelementptr inbounds bfloat, bfloat addrspace(1)* %30, i64 2
  %38 = load bfloat, bfloat addrspace(1)* %37, align 2, !tbaa !56
  %39 = fpext bfloat %38 to float
  %40 = fadd float %36, %39
  %41 = getelementptr inbounds bfloat, bfloat addrspace(1)* %30, i64 3
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !56
  %43 = fpext bfloat %42 to float
  %44 = fadd float %40, %43
  %45 = fadd float %44, 0.000000e+00
  store float %32, float* %21, align 4, !tbaa !67
  %46 = fmul float %35, 1.562500e-02
  store float %46, float* %22, align 4, !tbaa !67
  %47 = fmul float %39, 6.250000e-02
  store float %47, float* %23, align 4, !tbaa !67
  %48 = fmul float %43, 2.500000e-01
  store float %48, float* %24, align 4, !tbaa !67
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt64ELt4EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %28, i32 noundef %8, float* noundef nonnull %21, float noundef %45, i32 noundef 4, i1 noundef zeroext false, float* noundef nonnull %25) #10
  %49 = add i32 %27, 128
  %50 = icmp ugt i32 %17, %49
  %51 = sub i32 %17, %49
  %52 = icmp ugt i32 %51, 128
  %53 = and i1 %50, %52
  br i1 %53, label %26, label %54, !llvm.loop !111

54:                                               ; preds = %26, %11
  %55 = phi i32 [ 0, %11 ], [ %49, %26 ]
  %56 = phi i32 [ %17, %11 ], [ %51, %26 ]
  %57 = icmp ugt i32 %56, %19
  br i1 %57, label %58, label %61

58:                                               ; preds = %54
  %59 = sub i32 %56, %19
  %60 = call i32 @air.min.u.i32(i32 %59, i32 4) #15
  br label %61

61:                                               ; preds = %58, %54
  %62 = phi i32 [ %60, %58 ], [ 0, %54 ]
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %67

64:                                               ; preds = %61
  %65 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %66 = load i32, i32 addrspace(2)* %65, align 4, !tbaa !47
  br label %165

67:                                               ; preds = %61
  %68 = add i32 %55, %19
  %69 = zext i32 %68 to i64
  %70 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %69
  %71 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 0
  %72 = icmp sgt i32 %62, 0
  br i1 %72, label %76, label %73

73:                                               ; preds = %76, %67
  %74 = phi float [ 0.000000e+00, %67 ], [ %101, %76 ]
  %75 = icmp slt i32 %62, 4
  br i1 %75, label %111, label %117

76:                                               ; preds = %76, %67
  %77 = phi i32 [ %109, %76 ], [ 0, %67 ]
  %78 = phi float [ %101, %76 ], [ 0.000000e+00, %67 ]
  %79 = zext i32 %77 to i64
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %79
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !56
  %82 = fpext bfloat %81 to float
  %83 = or i32 %77, 1
  %84 = zext i32 %83 to i64
  %85 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %84
  %86 = load bfloat, bfloat addrspace(1)* %85, align 2, !tbaa !56
  %87 = fpext bfloat %86 to float
  %88 = fadd float %82, %87
  %89 = or i32 %77, 2
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %90
  %92 = load bfloat, bfloat addrspace(1)* %91, align 2, !tbaa !56
  %93 = fpext bfloat %92 to float
  %94 = fadd float %88, %93
  %95 = or i32 %77, 3
  %96 = zext i32 %95 to i64
  %97 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %96
  %98 = load bfloat, bfloat addrspace(1)* %97, align 2, !tbaa !56
  %99 = fpext bfloat %98 to float
  %100 = fadd float %94, %99
  %101 = fadd float %78, %100
  %102 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %79
  store float %82, float* %102, align 4, !tbaa !67
  %103 = fmul float %87, 1.562500e-02
  %104 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %84
  store float %103, float* %104, align 4, !tbaa !67
  %105 = fmul float %93, 6.250000e-02
  %106 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %90
  store float %105, float* %106, align 4, !tbaa !67
  %107 = fmul float %99, 2.500000e-01
  %108 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %96
  store float %107, float* %108, align 4, !tbaa !67
  %109 = add nuw nsw i32 %77, 4
  %110 = icmp slt i32 %109, %62
  br i1 %110, label %76, label %73, !llvm.loop !112

111:                                              ; preds = %111, %73
  %112 = phi i32 [ %115, %111 ], [ %62, %73 ]
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %113
  store float 0.000000e+00, float* %114, align 4, !tbaa !67
  %115 = add i32 %112, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %117, label %111, !llvm.loop !113

117:                                              ; preds = %111, %73
  %118 = zext i32 %8 to i64
  %119 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %120 = load i64, i64 addrspace(2)* %119, align 8, !tbaa !54
  %121 = mul i64 %120, %118
  %122 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %123 = load i64, i64 addrspace(2)* %122, align 8, !tbaa !77
  %124 = mul i64 %123, %118
  %125 = lshr i32 %68, 6
  %126 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %127 = load i32, i32 addrspace(2)* %126, align 4, !tbaa !47
  %128 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %124
  %129 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %130 = load i64, i64 addrspace(2)* %129, align 8
  %131 = mul nuw nsw i64 %69, 6
  %132 = lshr exact i64 %131, 3
  %133 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %134 = load i64, i64 addrspace(2)* %133, align 8
  %135 = zext i32 %125 to i64
  %136 = getelementptr inbounds i8, i8 addrspace(1)* %128, i64 %132
  br label %137

137:                                              ; preds = %162, %117
  %138 = phi i32 [ 0, %117 ], [ %163, %162 ]
  %139 = add i32 %138, %7
  %140 = icmp ult i32 %139, %127
  br i1 %140, label %141, label %162

141:                                              ; preds = %137
  %142 = zext i32 %139 to i64
  %143 = mul i64 %130, %142
  %144 = getelementptr inbounds i8, i8 addrspace(1)* %136, i64 %143
  %145 = mul i64 %134, %142
  %146 = add i64 %145, %121
  %147 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %146
  %148 = bitcast i8 addrspace(1)* %147 to bfloat addrspace(1)*
  %149 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %146
  %150 = bitcast i8 addrspace(1)* %149 to bfloat addrspace(1)*
  %151 = getelementptr inbounds bfloat, bfloat addrspace(1)* %148, i64 %135
  %152 = load bfloat, bfloat addrspace(1)* %151, align 2, !tbaa !56
  %153 = fpext bfloat %152 to float
  %154 = getelementptr inbounds bfloat, bfloat addrspace(1)* %150, i64 %135
  %155 = load bfloat, bfloat addrspace(1)* %154, align 2, !tbaa !56
  %156 = fpext bfloat %155 to float
  %157 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %144, float* noundef nonnull %71, float noundef %153, float noundef %156, float noundef %74, i32 noundef %62) #16
  %158 = zext i32 %138 to i64
  %159 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %158
  %160 = load float, float* %159, align 4, !tbaa !67
  %161 = fadd float %157, %160
  store float %161, float* %159, align 4, !tbaa !67
  br label %162

162:                                              ; preds = %141, %137
  %163 = add nuw nsw i32 %138, 1
  %164 = icmp eq i32 %163, 4
  br i1 %164, label %165, label %137, !llvm.loop !114

165:                                              ; preds = %162, %64
  %166 = phi i32 [ %66, %64 ], [ %127, %162 ]
  %167 = icmp eq i32 %10, 0
  %168 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %169 = zext i32 %166 to i64
  %170 = mul i64 %169, %9
  %171 = zext i32 %7 to i64
  %172 = add i64 %170, %171
  br label %174

173:                                              ; preds = %198
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %14) #12
  ret void

174:                                              ; preds = %198, %165
  %175 = phi i32 [ 0, %165 ], [ %199, %198 ]
  %176 = add i32 %175, %7
  %177 = icmp ult i32 %176, %166
  br i1 %177, label %178, label %198

178:                                              ; preds = %174
  %179 = zext i32 %175 to i64
  %180 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %179
  %181 = load float, float* %180, align 4, !tbaa !67
  %182 = call fast float @air.simd_sum.f32(float %181) #14
  br i1 %167, label %183, label %198

183:                                              ; preds = %178
  %184 = fptrunc float %182 to bfloat
  %185 = bitcast float %182 to i32
  %186 = and i32 %185, 2139095040
  %187 = icmp eq i32 %186, 2139095040
  br i1 %187, label %193, label %188

188:                                              ; preds = %183
  %189 = fpext bfloat %184 to float
  %190 = bitcast float %189 to i32
  %191 = and i32 %190, 2139095040
  %192 = icmp eq i32 %191, 2139095040
  br i1 %192, label %193, label %195

193:                                              ; preds = %188, %183
  %194 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %168, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %195

195:                                              ; preds = %193, %188
  %196 = add i64 %172, %179
  %197 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %196
  store bfloat %184, bfloat addrspace(1)* %197, align 2, !tbaa !56
  br label %198

198:                                              ; preds = %195, %178, %174
  %199 = add nuw nsw i32 %175, 1
  %200 = icmp eq i32 %199, 4
  br i1 %200, label %173, label %174, !llvm.loop !115
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #3 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = zext i32 %5 to i64
  %27 = mul nuw nsw i64 %26, 6
  %28 = lshr i64 %27, 3
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %30 = load i64, i64 addrspace(2)* %29, align 8
  %31 = zext i32 %20 to i64
  %32 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %28
  br label %34

33:                                               ; preds = %114
  ret void

34:                                               ; preds = %114, %12
  %35 = phi i32 [ 0, %12 ], [ %115, %114 ]
  %36 = add i32 %35, %4
  %37 = icmp ult i32 %36, %22
  br i1 %37, label %38, label %114

38:                                               ; preds = %34
  %39 = zext i32 %36 to i64
  %40 = mul i64 %25, %39
  %41 = getelementptr inbounds i8, i8 addrspace(1)* %32, i64 %40
  %42 = mul i64 %30, %39
  %43 = add i64 %42, %16
  %44 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %43
  %45 = bitcast i8 addrspace(1)* %44 to bfloat addrspace(1)*
  %46 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %43
  %47 = bitcast i8 addrspace(1)* %46 to bfloat addrspace(1)*
  %48 = getelementptr inbounds bfloat, bfloat addrspace(1)* %45, i64 %31
  %49 = load bfloat, bfloat addrspace(1)* %48, align 2, !tbaa !56
  %50 = fpext bfloat %49 to float
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %47, i64 %31
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !56
  %53 = fpext bfloat %52 to float
  br i1 %10, label %54, label %60

54:                                               ; preds = %38
  %55 = tail call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %41, float* noundef %7, float noundef %50, float noundef %53, float noundef %8, i32 noundef %9) #13
  %56 = zext i32 %35 to i64
  %57 = getelementptr inbounds float, float* %11, i64 %56
  %58 = load float, float* %57, align 4, !tbaa !67
  %59 = fadd float %55, %58
  store float %59, float* %57, align 4, !tbaa !67
  br label %114

60:                                               ; preds = %60, %38
  %61 = phi i1 [ false, %60 ], [ true, %38 ]
  %62 = phi i32 [ 1, %60 ], [ 0, %38 ]
  %63 = phi float [ %106, %60 ], [ 0.000000e+00, %38 ]
  %64 = phi i8 addrspace(1)* [ %71, %60 ], [ %41, %38 ]
  %65 = phi float* [ %68, %60 ], [ %7, %38 ]
  %66 = shl nuw nsw i32 %62, 2
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds float, float* %65, i64 %67
  %69 = mul nuw nsw i32 %62, 3
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds i8, i8 addrspace(1)* %64, i64 %70
  %72 = load i8, i8 addrspace(1)* %71, align 1, !tbaa !78
  %73 = zext i8 %72 to i32
  %74 = and i32 %73, 63
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #15
  %76 = load float, float* %68, align 4, !tbaa !67
  %77 = tail call float @llvm.fmuladd.f32(float %75, float %76, float %63) #12
  %78 = and i32 %73, 192
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #15
  %80 = getelementptr inbounds float, float* %68, i64 1
  %81 = load float, float* %80, align 4, !tbaa !67
  %82 = tail call float @llvm.fmuladd.f32(float %79, float %81, float %77) #12
  %83 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 1
  %84 = load i8, i8 addrspace(1)* %83, align 1, !tbaa !78
  %85 = zext i8 %84 to i32
  %86 = and i32 %85, 15
  %87 = tail call float @air.convert.f.f32.s.i32(i32 %86) #15
  %88 = fmul float %81, 2.560000e+02
  %89 = tail call float @llvm.fmuladd.f32(float %87, float %88, float %82) #12
  %90 = and i32 %85, 240
  %91 = tail call float @air.convert.f.f32.s.i32(i32 %90) #15
  %92 = getelementptr inbounds float, float* %68, i64 2
  %93 = load float, float* %92, align 4, !tbaa !67
  %94 = tail call float @llvm.fmuladd.f32(float %91, float %93, float %89) #12
  %95 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 2
  %96 = load i8, i8 addrspace(1)* %95, align 1, !tbaa !78
  %97 = zext i8 %96 to i32
  %98 = and i32 %97, 3
  %99 = tail call float @air.convert.f.f32.s.i32(i32 %98) #15
  %100 = fmul float %93, 2.560000e+02
  %101 = tail call float @llvm.fmuladd.f32(float %99, float %100, float %94) #12
  %102 = and i32 %97, 252
  %103 = tail call float @air.convert.f.f32.s.i32(i32 %102) #15
  %104 = getelementptr inbounds float, float* %68, i64 3
  %105 = load float, float* %104, align 4, !tbaa !67
  %106 = tail call float @llvm.fmuladd.f32(float %103, float %105, float %101) #12
  br i1 %61, label %60, label %107, !llvm.loop !116

107:                                              ; preds = %60
  %108 = fmul float %53, %8
  %109 = tail call float @llvm.fmuladd.f32(float %50, float %106, float %108) #12
  %110 = zext i32 %35 to i64
  %111 = getelementptr inbounds float, float* %11, i64 %110
  %112 = load float, float* %111, align 4, !tbaa !67
  %113 = fadd float %112, %109
  store float %113, float* %111, align 4, !tbaa !67
  br label %114

114:                                              ; preds = %107, %54, %34
  %115 = add nuw nsw i32 %35, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %33, label %34, !llvm.loop !117
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #6 {
  %7 = sdiv i32 %5, 4
  %8 = icmp sgt i32 %5, 3
  br i1 %8, label %13, label %9

9:                                                ; preds = %13, %6
  %10 = phi float [ 0.000000e+00, %6 ], [ %58, %13 ]
  %11 = fmul float %3, %4
  %12 = tail call float @llvm.fmuladd.f32(float %2, float %10, float %11)
  ret float %12

13:                                               ; preds = %13, %6
  %14 = phi i32 [ %59, %13 ], [ 0, %6 ]
  %15 = phi float [ %58, %13 ], [ 0.000000e+00, %6 ]
  %16 = phi i8 addrspace(1)* [ %23, %13 ], [ %0, %6 ]
  %17 = phi float* [ %20, %13 ], [ %1, %6 ]
  %18 = shl nsw i32 %14, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds float, float* %17, i64 %19
  %21 = mul nuw nsw i32 %14, 3
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %16, i64 %22
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !78
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 63
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #15
  %28 = load float, float* %20, align 4, !tbaa !67
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 192
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #15
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !67
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !78
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 15
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #15
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 240
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #15
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !67
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %48 = load i8, i8 addrspace(1)* %47, align 1, !tbaa !78
  %49 = zext i8 %48 to i32
  %50 = and i32 %49, 3
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %50) #15
  %52 = fmul float %45, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %46)
  %54 = and i32 %49, 252
  %55 = tail call float @air.convert.f.f32.s.i32(i32 %54) #15
  %56 = getelementptr inbounds float, float* %20, i64 3
  %57 = load float, float* %56, align 4, !tbaa !67
  %58 = tail call float @llvm.fmuladd.f32(float %55, float %57, float %53)
  %59 = add nuw nsw i32 %14, 1
  %60 = icmp eq i32 %59, %7
  br i1 %60, label %9, label %13, !llvm.loop !118
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt64ELt4EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #3 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = zext i32 %5 to i64
  %27 = mul nuw nsw i64 %26, 6
  %28 = lshr i64 %27, 3
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %30 = load i64, i64 addrspace(2)* %29, align 8
  %31 = zext i32 %20 to i64
  %32 = getelementptr inbounds float, float* %7, i64 1
  %33 = getelementptr inbounds float, float* %7, i64 2
  %34 = getelementptr inbounds float, float* %7, i64 3
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %28
  br label %37

36:                                               ; preds = %102
  ret void

37:                                               ; preds = %102, %12
  %38 = phi i32 [ 0, %12 ], [ %103, %102 ]
  %39 = add i32 %38, %4
  %40 = icmp ult i32 %39, %22
  br i1 %40, label %41, label %102

41:                                               ; preds = %37
  %42 = zext i32 %39 to i64
  %43 = mul i64 %25, %42
  %44 = getelementptr inbounds i8, i8 addrspace(1)* %35, i64 %43
  %45 = mul i64 %30, %42
  %46 = add i64 %45, %16
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %46
  %48 = bitcast i8 addrspace(1)* %47 to bfloat addrspace(1)*
  %49 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %46
  %50 = bitcast i8 addrspace(1)* %49 to bfloat addrspace(1)*
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %31
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !56
  %53 = fpext bfloat %52 to float
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %50, i64 %31
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !56
  %56 = fpext bfloat %55 to float
  br i1 %10, label %57, label %63

57:                                               ; preds = %41
  %58 = tail call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %44, float* noundef %7, float noundef %53, float noundef %56, float noundef %8, i32 noundef %9) #13
  %59 = zext i32 %38 to i64
  %60 = getelementptr inbounds float, float* %11, i64 %59
  %61 = load float, float* %60, align 4, !tbaa !67
  %62 = fadd float %58, %61
  store float %62, float* %60, align 4, !tbaa !67
  br label %102

63:                                               ; preds = %41
  %64 = load i8, i8 addrspace(1)* %44, align 1, !tbaa !78
  %65 = zext i8 %64 to i32
  %66 = and i32 %65, 63
  %67 = tail call float @air.convert.f.f32.s.i32(i32 %66) #15
  %68 = load float, float* %7, align 4, !tbaa !67
  %69 = and i32 %65, 192
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %69) #15
  %71 = load float, float* %32, align 4, !tbaa !67
  %72 = getelementptr inbounds i8, i8 addrspace(1)* %44, i64 1
  %73 = load i8, i8 addrspace(1)* %72, align 1, !tbaa !78
  %74 = zext i8 %73 to i32
  %75 = and i32 %74, 15
  %76 = tail call float @air.convert.f.f32.s.i32(i32 %75) #15
  %77 = fmul float %71, 2.560000e+02
  %78 = and i32 %74, 240
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #15
  %80 = load float, float* %33, align 4, !tbaa !67
  %81 = getelementptr inbounds i8, i8 addrspace(1)* %44, i64 2
  %82 = load i8, i8 addrspace(1)* %81, align 1, !tbaa !78
  %83 = zext i8 %82 to i32
  %84 = and i32 %83, 3
  %85 = tail call float @air.convert.f.f32.s.i32(i32 %84) #15
  %86 = fmul float %80, 2.560000e+02
  %87 = and i32 %83, 252
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %87) #15
  %89 = load float, float* %34, align 4, !tbaa !67
  %90 = tail call float @llvm.fmuladd.f32(float %67, float %68, float 0.000000e+00) #12
  %91 = tail call float @llvm.fmuladd.f32(float %70, float %71, float %90) #12
  %92 = tail call float @llvm.fmuladd.f32(float %76, float %77, float %91) #12
  %93 = tail call float @llvm.fmuladd.f32(float %79, float %80, float %92) #12
  %94 = tail call float @llvm.fmuladd.f32(float %85, float %86, float %93) #12
  %95 = tail call float @llvm.fmuladd.f32(float %88, float %89, float %94) #12
  %96 = fmul float %56, %8
  %97 = tail call float @llvm.fmuladd.f32(float %53, float %95, float %96) #12
  %98 = zext i32 %38 to i64
  %99 = getelementptr inbounds float, float* %11, i64 %98
  %100 = load float, float* %99, align 4, !tbaa !67
  %101 = fadd float %100, %97
  store float %101, float* %99, align 4, !tbaa !67
  br label %102

102:                                              ; preds = %63, %57, %37
  %103 = add nuw nsw i32 %38, 1
  %104 = icmp eq i32 %103, 4
  br i1 %104, label %36, label %37, !llvm.loop !114
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #6 {
  %7 = sdiv i32 %5, 4
  %8 = icmp sgt i32 %5, 3
  br i1 %8, label %13, label %9

9:                                                ; preds = %13, %6
  %10 = phi float [ 0.000000e+00, %6 ], [ %58, %13 ]
  %11 = fmul float %3, %4
  %12 = tail call float @llvm.fmuladd.f32(float %2, float %10, float %11)
  ret float %12

13:                                               ; preds = %13, %6
  %14 = phi i32 [ %59, %13 ], [ 0, %6 ]
  %15 = phi float [ %58, %13 ], [ 0.000000e+00, %6 ]
  %16 = phi i8 addrspace(1)* [ %23, %13 ], [ %0, %6 ]
  %17 = phi float* [ %20, %13 ], [ %1, %6 ]
  %18 = shl nsw i32 %14, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds float, float* %17, i64 %19
  %21 = mul nuw nsw i32 %14, 3
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %16, i64 %22
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !78
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 63
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #15
  %28 = load float, float* %20, align 4, !tbaa !67
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 192
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #15
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !67
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !78
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 15
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #15
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 240
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #15
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !67
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %48 = load i8, i8 addrspace(1)* %47, align 1, !tbaa !78
  %49 = zext i8 %48 to i32
  %50 = and i32 %49, 3
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %50) #15
  %52 = fmul float %45, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %46)
  %54 = and i32 %49, 252
  %55 = tail call float @air.convert.f.f32.s.i32(i32 %54) #15
  %56 = getelementptr inbounds float, float* %20, i64 3
  %57 = load float, float* %56, align 4, !tbaa !67
  %58 = tail call float @llvm.fmuladd.f32(float %55, float %57, float %53)
  %59 = add nuw nsw i32 %14, 1
  %60 = icmp eq i32 %59, %7
  br i1 %60, label %9, label %13, !llvm.loop !119
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %23, label %19

19:                                               ; preds = %11
  %20 = shl i32 %10, 3
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %32

23:                                               ; preds = %71, %11
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !47
  %26 = icmp eq i32 %10, 0
  %27 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %28 = zext i32 %25 to i64
  %29 = mul i64 %28, %9
  %30 = zext i32 %7 to i64
  %31 = add i64 %29, %30
  br label %75

32:                                               ; preds = %71, %19
  %33 = phi i32 [ 0, %19 ], [ %72, %71 ]
  %34 = add i32 %33, %20
  %35 = zext i32 %34 to i64
  %36 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %35
  br label %37

37:                                               ; preds = %37, %32
  %38 = phi i1 [ true, %32 ], [ false, %37 ]
  %39 = phi i32 [ 0, %32 ], [ 4, %37 ]
  %40 = phi float [ 0.000000e+00, %32 ], [ %63, %37 ]
  %41 = zext i32 %39 to i64
  %42 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %41
  %43 = load bfloat, bfloat addrspace(1)* %42, align 2, !tbaa !56
  %44 = fpext bfloat %43 to float
  %45 = or i32 %39, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !56
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %39, 2
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !56
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = or i32 %39, 3
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %58
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !56
  %61 = fpext bfloat %60 to float
  %62 = fadd float %56, %61
  %63 = fadd float %40, %62
  %64 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %41
  store float %44, float* %64, align 4, !tbaa !67
  %65 = fmul float %49, 1.562500e-02
  %66 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %46
  store float %65, float* %66, align 4, !tbaa !67
  %67 = fmul float %55, 6.250000e-02
  %68 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %52
  store float %67, float* %68, align 4, !tbaa !67
  %69 = fmul float %61, 2.500000e-01
  %70 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %58
  store float %69, float* %70, align 4, !tbaa !67
  br i1 %38, label %37, label %71, !llvm.loop !108

71:                                               ; preds = %37
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt128ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %34, i32 noundef %8, float* noundef nonnull %21, float noundef %63, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %22) #10
  %72 = add i32 %33, 256
  %73 = icmp ult i32 %72, %17
  br i1 %73, label %32, label %23, !llvm.loop !120

74:                                               ; preds = %99
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

75:                                               ; preds = %99, %23
  %76 = phi i32 [ 0, %23 ], [ %100, %99 ]
  %77 = add i32 %76, %7
  %78 = icmp ult i32 %77, %25
  br i1 %78, label %79, label %99

79:                                               ; preds = %75
  %80 = zext i32 %76 to i64
  %81 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %80
  %82 = load float, float* %81, align 4, !tbaa !67
  %83 = call fast float @air.simd_sum.f32(float %82) #14
  br i1 %26, label %84, label %99

84:                                               ; preds = %79
  %85 = fptrunc float %83 to bfloat
  %86 = bitcast float %83 to i32
  %87 = and i32 %86, 2139095040
  %88 = icmp eq i32 %87, 2139095040
  br i1 %88, label %94, label %89

89:                                               ; preds = %84
  %90 = fpext bfloat %85 to float
  %91 = bitcast float %90 to i32
  %92 = and i32 %91, 2139095040
  %93 = icmp eq i32 %92, 2139095040
  br i1 %93, label %94, label %96

94:                                               ; preds = %89, %84
  %95 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %27, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %96

96:                                               ; preds = %94, %89
  %97 = add i64 %31, %80
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %97
  store bfloat %85, bfloat addrspace(1)* %98, align 2, !tbaa !56
  br label %99

99:                                               ; preds = %96, %79, %75
  %100 = add nuw nsw i32 %76, 1
  %101 = icmp eq i32 %100, 4
  br i1 %101, label %74, label %75, !llvm.loop !121
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt6ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [4 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [4 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 128
  %19 = shl i32 %10, 2
  br i1 %18, label %20, label %54

20:                                               ; preds = %11
  %21 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 1
  %23 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 2
  %24 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 3
  %25 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %26

26:                                               ; preds = %26, %20
  %27 = phi i32 [ 0, %20 ], [ %49, %26 ]
  %28 = add i32 %27, %19
  %29 = zext i32 %28 to i64
  %30 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %29
  %31 = load bfloat, bfloat addrspace(1)* %30, align 2, !tbaa !56
  %32 = fpext bfloat %31 to float
  %33 = getelementptr inbounds bfloat, bfloat addrspace(1)* %30, i64 1
  %34 = load bfloat, bfloat addrspace(1)* %33, align 2, !tbaa !56
  %35 = fpext bfloat %34 to float
  %36 = fadd float %32, %35
  %37 = getelementptr inbounds bfloat, bfloat addrspace(1)* %30, i64 2
  %38 = load bfloat, bfloat addrspace(1)* %37, align 2, !tbaa !56
  %39 = fpext bfloat %38 to float
  %40 = fadd float %36, %39
  %41 = getelementptr inbounds bfloat, bfloat addrspace(1)* %30, i64 3
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !56
  %43 = fpext bfloat %42 to float
  %44 = fadd float %40, %43
  %45 = fadd float %44, 0.000000e+00
  store float %32, float* %21, align 4, !tbaa !67
  %46 = fmul float %35, 1.562500e-02
  store float %46, float* %22, align 4, !tbaa !67
  %47 = fmul float %39, 6.250000e-02
  store float %47, float* %23, align 4, !tbaa !67
  %48 = fmul float %43, 2.500000e-01
  store float %48, float* %24, align 4, !tbaa !67
  call void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt128ELt4EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %28, i32 noundef %8, float* noundef nonnull %21, float noundef %45, i32 noundef 4, i1 noundef zeroext false, float* noundef nonnull %25) #10
  %49 = add i32 %27, 128
  %50 = icmp ugt i32 %17, %49
  %51 = sub i32 %17, %49
  %52 = icmp ugt i32 %51, 128
  %53 = and i1 %50, %52
  br i1 %53, label %26, label %54, !llvm.loop !122

54:                                               ; preds = %26, %11
  %55 = phi i32 [ 0, %11 ], [ %49, %26 ]
  %56 = phi i32 [ %17, %11 ], [ %51, %26 ]
  %57 = icmp ugt i32 %56, %19
  br i1 %57, label %58, label %61

58:                                               ; preds = %54
  %59 = sub i32 %56, %19
  %60 = call i32 @air.min.u.i32(i32 %59, i32 4) #15
  br label %61

61:                                               ; preds = %58, %54
  %62 = phi i32 [ %60, %58 ], [ 0, %54 ]
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %64, label %67

64:                                               ; preds = %61
  %65 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %66 = load i32, i32 addrspace(2)* %65, align 4, !tbaa !47
  br label %165

67:                                               ; preds = %61
  %68 = add i32 %55, %19
  %69 = zext i32 %68 to i64
  %70 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %69
  %71 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 0
  %72 = icmp sgt i32 %62, 0
  br i1 %72, label %76, label %73

73:                                               ; preds = %76, %67
  %74 = phi float [ 0.000000e+00, %67 ], [ %101, %76 ]
  %75 = icmp slt i32 %62, 4
  br i1 %75, label %111, label %117

76:                                               ; preds = %76, %67
  %77 = phi i32 [ %109, %76 ], [ 0, %67 ]
  %78 = phi float [ %101, %76 ], [ 0.000000e+00, %67 ]
  %79 = zext i32 %77 to i64
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %79
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !56
  %82 = fpext bfloat %81 to float
  %83 = or i32 %77, 1
  %84 = zext i32 %83 to i64
  %85 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %84
  %86 = load bfloat, bfloat addrspace(1)* %85, align 2, !tbaa !56
  %87 = fpext bfloat %86 to float
  %88 = fadd float %82, %87
  %89 = or i32 %77, 2
  %90 = zext i32 %89 to i64
  %91 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %90
  %92 = load bfloat, bfloat addrspace(1)* %91, align 2, !tbaa !56
  %93 = fpext bfloat %92 to float
  %94 = fadd float %88, %93
  %95 = or i32 %77, 3
  %96 = zext i32 %95 to i64
  %97 = getelementptr inbounds bfloat, bfloat addrspace(1)* %70, i64 %96
  %98 = load bfloat, bfloat addrspace(1)* %97, align 2, !tbaa !56
  %99 = fpext bfloat %98 to float
  %100 = fadd float %94, %99
  %101 = fadd float %78, %100
  %102 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %79
  store float %82, float* %102, align 4, !tbaa !67
  %103 = fmul float %87, 1.562500e-02
  %104 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %84
  store float %103, float* %104, align 4, !tbaa !67
  %105 = fmul float %93, 6.250000e-02
  %106 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %90
  store float %105, float* %106, align 4, !tbaa !67
  %107 = fmul float %99, 2.500000e-01
  %108 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %96
  store float %107, float* %108, align 4, !tbaa !67
  %109 = add nuw nsw i32 %77, 4
  %110 = icmp slt i32 %109, %62
  br i1 %110, label %76, label %73, !llvm.loop !112

111:                                              ; preds = %111, %73
  %112 = phi i32 [ %115, %111 ], [ %62, %73 ]
  %113 = sext i32 %112 to i64
  %114 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %113
  store float 0.000000e+00, float* %114, align 4, !tbaa !67
  %115 = add i32 %112, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %117, label %111, !llvm.loop !113

117:                                              ; preds = %111, %73
  %118 = zext i32 %8 to i64
  %119 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %120 = load i64, i64 addrspace(2)* %119, align 8, !tbaa !54
  %121 = mul i64 %120, %118
  %122 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %123 = load i64, i64 addrspace(2)* %122, align 8, !tbaa !77
  %124 = mul i64 %123, %118
  %125 = lshr i32 %68, 7
  %126 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %127 = load i32, i32 addrspace(2)* %126, align 4, !tbaa !47
  %128 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %124
  %129 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %130 = load i64, i64 addrspace(2)* %129, align 8
  %131 = mul nuw nsw i64 %69, 6
  %132 = lshr exact i64 %131, 3
  %133 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %134 = load i64, i64 addrspace(2)* %133, align 8
  %135 = zext i32 %125 to i64
  %136 = getelementptr inbounds i8, i8 addrspace(1)* %128, i64 %132
  br label %137

137:                                              ; preds = %162, %117
  %138 = phi i32 [ 0, %117 ], [ %163, %162 ]
  %139 = add i32 %138, %7
  %140 = icmp ult i32 %139, %127
  br i1 %140, label %141, label %162

141:                                              ; preds = %137
  %142 = zext i32 %139 to i64
  %143 = mul i64 %130, %142
  %144 = getelementptr inbounds i8, i8 addrspace(1)* %136, i64 %143
  %145 = mul i64 %134, %142
  %146 = add i64 %145, %121
  %147 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %146
  %148 = bitcast i8 addrspace(1)* %147 to bfloat addrspace(1)*
  %149 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %146
  %150 = bitcast i8 addrspace(1)* %149 to bfloat addrspace(1)*
  %151 = getelementptr inbounds bfloat, bfloat addrspace(1)* %148, i64 %135
  %152 = load bfloat, bfloat addrspace(1)* %151, align 2, !tbaa !56
  %153 = fpext bfloat %152 to float
  %154 = getelementptr inbounds bfloat, bfloat addrspace(1)* %150, i64 %135
  %155 = load bfloat, bfloat addrspace(1)* %154, align 2, !tbaa !56
  %156 = fpext bfloat %155 to float
  %157 = call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %144, float* noundef nonnull %71, float noundef %153, float noundef %156, float noundef %74, i32 noundef %62) #16
  %158 = zext i32 %138 to i64
  %159 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %158
  %160 = load float, float* %159, align 4, !tbaa !67
  %161 = fadd float %157, %160
  store float %161, float* %159, align 4, !tbaa !67
  br label %162

162:                                              ; preds = %141, %137
  %163 = add nuw nsw i32 %138, 1
  %164 = icmp eq i32 %163, 4
  br i1 %164, label %165, label %137, !llvm.loop !123

165:                                              ; preds = %162, %64
  %166 = phi i32 [ %66, %64 ], [ %127, %162 ]
  %167 = icmp eq i32 %10, 0
  %168 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %169 = zext i32 %166 to i64
  %170 = mul i64 %169, %9
  %171 = zext i32 %7 to i64
  %172 = add i64 %170, %171
  br label %174

173:                                              ; preds = %198
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %14) #12
  ret void

174:                                              ; preds = %198, %165
  %175 = phi i32 [ 0, %165 ], [ %199, %198 ]
  %176 = add i32 %175, %7
  %177 = icmp ult i32 %176, %166
  br i1 %177, label %178, label %198

178:                                              ; preds = %174
  %179 = zext i32 %175 to i64
  %180 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %179
  %181 = load float, float* %180, align 4, !tbaa !67
  %182 = call fast float @air.simd_sum.f32(float %181) #14
  br i1 %167, label %183, label %198

183:                                              ; preds = %178
  %184 = fptrunc float %182 to bfloat
  %185 = bitcast float %182 to i32
  %186 = and i32 %185, 2139095040
  %187 = icmp eq i32 %186, 2139095040
  br i1 %187, label %193, label %188

188:                                              ; preds = %183
  %189 = fpext bfloat %184 to float
  %190 = bitcast float %189 to i32
  %191 = and i32 %190, 2139095040
  %192 = icmp eq i32 %191, 2139095040
  br i1 %192, label %193, label %195

193:                                              ; preds = %188, %183
  %194 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %168, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %195

195:                                              ; preds = %193, %188
  %196 = add i64 %172, %179
  %197 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %196
  store bfloat %184, bfloat addrspace(1)* %197, align 2, !tbaa !56
  br label %198

198:                                              ; preds = %195, %178, %174
  %199 = add nuw nsw i32 %175, 1
  %200 = icmp eq i32 %199, 4
  br i1 %200, label %173, label %174, !llvm.loop !124
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt128ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #3 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 7
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = zext i32 %5 to i64
  %27 = mul nuw nsw i64 %26, 6
  %28 = lshr i64 %27, 3
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %30 = load i64, i64 addrspace(2)* %29, align 8
  %31 = zext i32 %20 to i64
  %32 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %28
  br label %34

33:                                               ; preds = %114
  ret void

34:                                               ; preds = %114, %12
  %35 = phi i32 [ 0, %12 ], [ %115, %114 ]
  %36 = add i32 %35, %4
  %37 = icmp ult i32 %36, %22
  br i1 %37, label %38, label %114

38:                                               ; preds = %34
  %39 = zext i32 %36 to i64
  %40 = mul i64 %25, %39
  %41 = getelementptr inbounds i8, i8 addrspace(1)* %32, i64 %40
  %42 = mul i64 %30, %39
  %43 = add i64 %42, %16
  %44 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %43
  %45 = bitcast i8 addrspace(1)* %44 to bfloat addrspace(1)*
  %46 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %43
  %47 = bitcast i8 addrspace(1)* %46 to bfloat addrspace(1)*
  %48 = getelementptr inbounds bfloat, bfloat addrspace(1)* %45, i64 %31
  %49 = load bfloat, bfloat addrspace(1)* %48, align 2, !tbaa !56
  %50 = fpext bfloat %49 to float
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %47, i64 %31
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !56
  %53 = fpext bfloat %52 to float
  br i1 %10, label %54, label %60

54:                                               ; preds = %38
  %55 = tail call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %41, float* noundef %7, float noundef %50, float noundef %53, float noundef %8, i32 noundef %9) #13
  %56 = zext i32 %35 to i64
  %57 = getelementptr inbounds float, float* %11, i64 %56
  %58 = load float, float* %57, align 4, !tbaa !67
  %59 = fadd float %55, %58
  store float %59, float* %57, align 4, !tbaa !67
  br label %114

60:                                               ; preds = %60, %38
  %61 = phi i1 [ false, %60 ], [ true, %38 ]
  %62 = phi i32 [ 1, %60 ], [ 0, %38 ]
  %63 = phi float [ %106, %60 ], [ 0.000000e+00, %38 ]
  %64 = phi i8 addrspace(1)* [ %71, %60 ], [ %41, %38 ]
  %65 = phi float* [ %68, %60 ], [ %7, %38 ]
  %66 = shl nuw nsw i32 %62, 2
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds float, float* %65, i64 %67
  %69 = mul nuw nsw i32 %62, 3
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds i8, i8 addrspace(1)* %64, i64 %70
  %72 = load i8, i8 addrspace(1)* %71, align 1, !tbaa !78
  %73 = zext i8 %72 to i32
  %74 = and i32 %73, 63
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #15
  %76 = load float, float* %68, align 4, !tbaa !67
  %77 = tail call float @llvm.fmuladd.f32(float %75, float %76, float %63) #12
  %78 = and i32 %73, 192
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #15
  %80 = getelementptr inbounds float, float* %68, i64 1
  %81 = load float, float* %80, align 4, !tbaa !67
  %82 = tail call float @llvm.fmuladd.f32(float %79, float %81, float %77) #12
  %83 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 1
  %84 = load i8, i8 addrspace(1)* %83, align 1, !tbaa !78
  %85 = zext i8 %84 to i32
  %86 = and i32 %85, 15
  %87 = tail call float @air.convert.f.f32.s.i32(i32 %86) #15
  %88 = fmul float %81, 2.560000e+02
  %89 = tail call float @llvm.fmuladd.f32(float %87, float %88, float %82) #12
  %90 = and i32 %85, 240
  %91 = tail call float @air.convert.f.f32.s.i32(i32 %90) #15
  %92 = getelementptr inbounds float, float* %68, i64 2
  %93 = load float, float* %92, align 4, !tbaa !67
  %94 = tail call float @llvm.fmuladd.f32(float %91, float %93, float %89) #12
  %95 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 2
  %96 = load i8, i8 addrspace(1)* %95, align 1, !tbaa !78
  %97 = zext i8 %96 to i32
  %98 = and i32 %97, 3
  %99 = tail call float @air.convert.f.f32.s.i32(i32 %98) #15
  %100 = fmul float %93, 2.560000e+02
  %101 = tail call float @llvm.fmuladd.f32(float %99, float %100, float %94) #12
  %102 = and i32 %97, 252
  %103 = tail call float @air.convert.f.f32.s.i32(i32 %102) #15
  %104 = getelementptr inbounds float, float* %68, i64 3
  %105 = load float, float* %104, align 4, !tbaa !67
  %106 = tail call float @llvm.fmuladd.f32(float %103, float %105, float %101) #12
  br i1 %61, label %60, label %107, !llvm.loop !116

107:                                              ; preds = %60
  %108 = fmul float %53, %8
  %109 = tail call float @llvm.fmuladd.f32(float %50, float %106, float %108) #12
  %110 = zext i32 %35 to i64
  %111 = getelementptr inbounds float, float* %11, i64 %110
  %112 = load float, float* %111, align 4, !tbaa !67
  %113 = fadd float %112, %109
  store float %113, float* %111, align 4, !tbaa !67
  br label %114

114:                                              ; preds = %107, %54, %34
  %115 = add nuw nsw i32 %35, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %33, label %34, !llvm.loop !125
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v116accumulate_chunkILt6ELt128ELt4EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #3 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !54
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !77
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 7
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !47
  %23 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %19
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8
  %26 = zext i32 %5 to i64
  %27 = mul nuw nsw i64 %26, 6
  %28 = lshr i64 %27, 3
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 10
  %30 = load i64, i64 addrspace(2)* %29, align 8
  %31 = zext i32 %20 to i64
  %32 = getelementptr inbounds float, float* %7, i64 1
  %33 = getelementptr inbounds float, float* %7, i64 2
  %34 = getelementptr inbounds float, float* %7, i64 3
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 %28
  br label %37

36:                                               ; preds = %102
  ret void

37:                                               ; preds = %102, %12
  %38 = phi i32 [ 0, %12 ], [ %103, %102 ]
  %39 = add i32 %38, %4
  %40 = icmp ult i32 %39, %22
  br i1 %40, label %41, label %102

41:                                               ; preds = %37
  %42 = zext i32 %39 to i64
  %43 = mul i64 %25, %42
  %44 = getelementptr inbounds i8, i8 addrspace(1)* %35, i64 %43
  %45 = mul i64 %30, %42
  %46 = add i64 %45, %16
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %46
  %48 = bitcast i8 addrspace(1)* %47 to bfloat addrspace(1)*
  %49 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %46
  %50 = bitcast i8 addrspace(1)* %49 to bfloat addrspace(1)*
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %31
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !56
  %53 = fpext bfloat %52 to float
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %50, i64 %31
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !56
  %56 = fpext bfloat %55 to float
  br i1 %10, label %57, label %63

57:                                               ; preds = %41
  %58 = tail call fast float @_ZN25splash_mlx_qmv_f32xsum_v128mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %44, float* noundef %7, float noundef %53, float noundef %56, float noundef %8, i32 noundef %9) #13
  %59 = zext i32 %38 to i64
  %60 = getelementptr inbounds float, float* %11, i64 %59
  %61 = load float, float* %60, align 4, !tbaa !67
  %62 = fadd float %58, %61
  store float %62, float* %60, align 4, !tbaa !67
  br label %102

63:                                               ; preds = %41
  %64 = load i8, i8 addrspace(1)* %44, align 1, !tbaa !78
  %65 = zext i8 %64 to i32
  %66 = and i32 %65, 63
  %67 = tail call float @air.convert.f.f32.s.i32(i32 %66) #15
  %68 = load float, float* %7, align 4, !tbaa !67
  %69 = and i32 %65, 192
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %69) #15
  %71 = load float, float* %32, align 4, !tbaa !67
  %72 = getelementptr inbounds i8, i8 addrspace(1)* %44, i64 1
  %73 = load i8, i8 addrspace(1)* %72, align 1, !tbaa !78
  %74 = zext i8 %73 to i32
  %75 = and i32 %74, 15
  %76 = tail call float @air.convert.f.f32.s.i32(i32 %75) #15
  %77 = fmul float %71, 2.560000e+02
  %78 = and i32 %74, 240
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #15
  %80 = load float, float* %33, align 4, !tbaa !67
  %81 = getelementptr inbounds i8, i8 addrspace(1)* %44, i64 2
  %82 = load i8, i8 addrspace(1)* %81, align 1, !tbaa !78
  %83 = zext i8 %82 to i32
  %84 = and i32 %83, 3
  %85 = tail call float @air.convert.f.f32.s.i32(i32 %84) #15
  %86 = fmul float %80, 2.560000e+02
  %87 = and i32 %83, 252
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %87) #15
  %89 = load float, float* %34, align 4, !tbaa !67
  %90 = tail call float @llvm.fmuladd.f32(float %67, float %68, float 0.000000e+00) #12
  %91 = tail call float @llvm.fmuladd.f32(float %70, float %71, float %90) #12
  %92 = tail call float @llvm.fmuladd.f32(float %76, float %77, float %91) #12
  %93 = tail call float @llvm.fmuladd.f32(float %79, float %80, float %92) #12
  %94 = tail call float @llvm.fmuladd.f32(float %85, float %86, float %93) #12
  %95 = tail call float @llvm.fmuladd.f32(float %88, float %89, float %94) #12
  %96 = fmul float %56, %8
  %97 = tail call float @llvm.fmuladd.f32(float %53, float %95, float %96) #12
  %98 = zext i32 %38 to i64
  %99 = getelementptr inbounds float, float* %11, i64 %98
  %100 = load float, float* %99, align 4, !tbaa !67
  %101 = fadd float %100, %97
  store float %101, float* %99, align 4, !tbaa !67
  br label %102

102:                                              ; preds = %63, %57, %37
  %103 = add nuw nsw i32 %38, 1
  %104 = icmp eq i32 %103, 4
  br i1 %104, label %36, label %37, !llvm.loop !123
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %11
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !47
  br label %38

22:                                               ; preds = %11
  %23 = shl i32 %10, 3
  %24 = zext i32 %8 to i64
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %26 = load i64, i64 addrspace(2)* %25, align 8, !tbaa !54
  %27 = mul i64 %26, %24
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %29 = load i64, i64 addrspace(2)* %28, align 8, !tbaa !77
  %30 = mul i64 %29, %24
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %32 = load i32, i32 addrspace(2)* %31, align 4, !tbaa !47
  %33 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %30
  %34 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %35 = load i64, i64 addrspace(2)* %34, align 8
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %37 = load i64, i64 addrspace(2)* %36, align 8
  br label %46

38:                                               ; preds = %109, %19
  %39 = phi i32 [ %21, %19 ], [ %32, %109 ]
  %40 = icmp eq i32 %10, 0
  %41 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %42 = zext i32 %39 to i64
  %43 = mul i64 %42, %9
  %44 = zext i32 %7 to i64
  %45 = add i64 %43, %44
  br label %113

46:                                               ; preds = %109, %22
  %47 = phi i32 [ 0, %22 ], [ %110, %109 ]
  %48 = add i32 %47, %23
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %49
  br label %51

51:                                               ; preds = %51, %46
  %52 = phi i32 [ 0, %46 ], [ %60, %51 ]
  %53 = phi float [ 0.000000e+00, %46 ], [ %58, %51 ]
  %54 = zext i32 %52 to i64
  %55 = getelementptr inbounds bfloat, bfloat addrspace(1)* %50, i64 %54
  %56 = load bfloat, bfloat addrspace(1)* %55, align 2, !tbaa !56
  %57 = fpext bfloat %56 to float
  %58 = fadd float %53, %57
  %59 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %54
  store float %57, float* %59, align 4, !tbaa !67
  %60 = add nuw nsw i32 %52, 1
  %61 = icmp eq i32 %60, 8
  br i1 %61, label %62, label %51, !llvm.loop !126

62:                                               ; preds = %51
  %63 = lshr i32 %48, 6
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds i8, i8 addrspace(1)* %33, i64 %49
  br label %66

66:                                               ; preds = %106, %62
  %67 = phi i32 [ 0, %62 ], [ %107, %106 ]
  %68 = add i32 %67, %7
  %69 = icmp ult i32 %68, %32
  br i1 %69, label %70, label %106

70:                                               ; preds = %66
  %71 = zext i32 %68 to i64
  %72 = mul i64 %35, %71
  %73 = getelementptr inbounds i8, i8 addrspace(1)* %65, i64 %72
  %74 = mul i64 %37, %71
  %75 = add i64 %74, %27
  %76 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %75
  %77 = bitcast i8 addrspace(1)* %76 to bfloat addrspace(1)*
  %78 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %75
  %79 = bitcast i8 addrspace(1)* %78 to bfloat addrspace(1)*
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %77, i64 %64
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !56
  %82 = getelementptr inbounds bfloat, bfloat addrspace(1)* %79, i64 %64
  %83 = load bfloat, bfloat addrspace(1)* %82, align 2, !tbaa !56
  br label %84

84:                                               ; preds = %84, %70
  %85 = phi i32 [ %95, %84 ], [ 0, %70 ]
  %86 = phi float [ %94, %84 ], [ 0.000000e+00, %70 ]
  %87 = zext i32 %85 to i64
  %88 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %87
  %89 = load float, float* %88, align 4, !tbaa !67
  %90 = getelementptr inbounds i8, i8 addrspace(1)* %73, i64 %87
  %91 = load i8, i8 addrspace(1)* %90, align 1, !tbaa !78
  %92 = zext i8 %91 to i32
  %93 = tail call float @air.convert.f.f32.s.i32(i32 %92) #15
  %94 = tail call float @llvm.fmuladd.f32(float %89, float %93, float %86) #12
  %95 = add nuw nsw i32 %85, 1
  %96 = icmp eq i32 %95, 8
  br i1 %96, label %97, label %84, !llvm.loop !127

97:                                               ; preds = %84
  %98 = fpext bfloat %81 to float
  %99 = fpext bfloat %83 to float
  %100 = fmul float %58, %99
  %101 = tail call float @llvm.fmuladd.f32(float %98, float %94, float %100) #12
  %102 = zext i32 %67 to i64
  %103 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %102
  %104 = load float, float* %103, align 4, !tbaa !67
  %105 = fadd float %104, %101
  store float %105, float* %103, align 4, !tbaa !67
  br label %106

106:                                              ; preds = %97, %66
  %107 = add nuw nsw i32 %67, 1
  %108 = icmp eq i32 %107, 4
  br i1 %108, label %109, label %66, !llvm.loop !128

109:                                              ; preds = %106
  %110 = add nuw i32 %47, 256
  %111 = icmp ult i32 %110, %17
  br i1 %111, label %46, label %38, !llvm.loop !129

112:                                              ; preds = %137
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

113:                                              ; preds = %137, %38
  %114 = phi i32 [ 0, %38 ], [ %138, %137 ]
  %115 = add i32 %114, %7
  %116 = icmp ult i32 %115, %39
  br i1 %116, label %117, label %137

117:                                              ; preds = %113
  %118 = zext i32 %114 to i64
  %119 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %118
  %120 = load float, float* %119, align 4, !tbaa !67
  %121 = tail call fast float @air.simd_sum.f32(float %120) #14
  br i1 %40, label %122, label %137

122:                                              ; preds = %117
  %123 = fptrunc float %121 to bfloat
  %124 = bitcast float %121 to i32
  %125 = and i32 %124, 2139095040
  %126 = icmp eq i32 %125, 2139095040
  br i1 %126, label %132, label %127

127:                                              ; preds = %122
  %128 = fpext bfloat %123 to float
  %129 = bitcast float %128 to i32
  %130 = and i32 %129, 2139095040
  %131 = icmp eq i32 %130, 2139095040
  br i1 %131, label %132, label %134

132:                                              ; preds = %127, %122
  %133 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %41, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %134

134:                                              ; preds = %132, %127
  %135 = add i64 %45, %118
  %136 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %135
  store bfloat %123, bfloat addrspace(1)* %136, align 2, !tbaa !56
  br label %137

137:                                              ; preds = %134, %117, %113
  %138 = add nuw nsw i32 %114, 1
  %139 = icmp eq i32 %138, 4
  br i1 %139, label %112, label %113, !llvm.loop !130
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [4 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [4 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 128
  %19 = shl i32 %10, 2
  br i1 %18, label %20, label %104

20:                                               ; preds = %11
  %21 = zext i32 %8 to i64
  %22 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %23 = load i64, i64 addrspace(2)* %22, align 8, !tbaa !54
  %24 = mul i64 %23, %21
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %26 = load i64, i64 addrspace(2)* %25, align 8, !tbaa !77
  %27 = mul i64 %26, %21
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %29 = load i32, i32 addrspace(2)* %28, align 4, !tbaa !47
  %30 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %27
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %32 = load i64, i64 addrspace(2)* %31, align 8
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %34 = load i64, i64 addrspace(2)* %33, align 8
  br label %35

35:                                               ; preds = %98, %20
  %36 = phi i32 [ 0, %20 ], [ %99, %98 ]
  %37 = add i32 %36, %19
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %38
  br label %40

40:                                               ; preds = %40, %35
  %41 = phi i32 [ 0, %35 ], [ %49, %40 ]
  %42 = phi float [ 0.000000e+00, %35 ], [ %47, %40 ]
  %43 = zext i32 %41 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %39, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !56
  %46 = fpext bfloat %45 to float
  %47 = fadd float %42, %46
  %48 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %43
  store float %46, float* %48, align 4, !tbaa !67
  %49 = add nuw nsw i32 %41, 1
  %50 = icmp eq i32 %49, 4
  br i1 %50, label %51, label %40, !llvm.loop !131

51:                                               ; preds = %40
  %52 = lshr i32 %37, 6
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds i8, i8 addrspace(1)* %30, i64 %38
  br label %55

55:                                               ; preds = %95, %51
  %56 = phi i32 [ 0, %51 ], [ %96, %95 ]
  %57 = add i32 %56, %7
  %58 = icmp ult i32 %57, %29
  br i1 %58, label %59, label %95

59:                                               ; preds = %55
  %60 = zext i32 %57 to i64
  %61 = mul i64 %32, %60
  %62 = getelementptr inbounds i8, i8 addrspace(1)* %54, i64 %61
  %63 = mul i64 %34, %60
  %64 = add i64 %63, %24
  %65 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %64
  %66 = bitcast i8 addrspace(1)* %65 to bfloat addrspace(1)*
  %67 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %64
  %68 = bitcast i8 addrspace(1)* %67 to bfloat addrspace(1)*
  %69 = getelementptr inbounds bfloat, bfloat addrspace(1)* %66, i64 %53
  %70 = load bfloat, bfloat addrspace(1)* %69, align 2, !tbaa !56
  %71 = getelementptr inbounds bfloat, bfloat addrspace(1)* %68, i64 %53
  %72 = load bfloat, bfloat addrspace(1)* %71, align 2, !tbaa !56
  br label %73

73:                                               ; preds = %73, %59
  %74 = phi i32 [ %84, %73 ], [ 0, %59 ]
  %75 = phi float [ %83, %73 ], [ 0.000000e+00, %59 ]
  %76 = zext i32 %74 to i64
  %77 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %76
  %78 = load float, float* %77, align 4, !tbaa !67
  %79 = getelementptr inbounds i8, i8 addrspace(1)* %62, i64 %76
  %80 = load i8, i8 addrspace(1)* %79, align 1, !tbaa !78
  %81 = zext i8 %80 to i32
  %82 = tail call float @air.convert.f.f32.s.i32(i32 %81) #15
  %83 = tail call float @llvm.fmuladd.f32(float %78, float %82, float %75) #12
  %84 = add nuw nsw i32 %74, 1
  %85 = icmp eq i32 %84, 4
  br i1 %85, label %86, label %73, !llvm.loop !132

86:                                               ; preds = %73
  %87 = fpext bfloat %70 to float
  %88 = fpext bfloat %72 to float
  %89 = fmul float %47, %88
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %83, float %89) #12
  %91 = zext i32 %56 to i64
  %92 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %91
  %93 = load float, float* %92, align 4, !tbaa !67
  %94 = fadd float %93, %90
  store float %94, float* %92, align 4, !tbaa !67
  br label %95

95:                                               ; preds = %86, %55
  %96 = add nuw nsw i32 %56, 1
  %97 = icmp eq i32 %96, 4
  br i1 %97, label %98, label %55, !llvm.loop !133

98:                                               ; preds = %95
  %99 = add i32 %36, 128
  %100 = icmp ugt i32 %17, %99
  %101 = sub i32 %17, %99
  %102 = icmp ugt i32 %101, 128
  %103 = and i1 %100, %102
  br i1 %103, label %35, label %104, !llvm.loop !134

104:                                              ; preds = %98, %11
  %105 = phi i32 [ 0, %11 ], [ %99, %98 ]
  %106 = phi i32 [ %17, %11 ], [ %101, %98 ]
  %107 = icmp ugt i32 %106, %19
  br i1 %107, label %108, label %111

108:                                              ; preds = %104
  %109 = sub i32 %106, %19
  %110 = tail call i32 @air.min.u.i32(i32 %109, i32 4) #15
  br label %111

111:                                              ; preds = %108, %104
  %112 = phi i32 [ %110, %108 ], [ 0, %104 ]
  %113 = icmp eq i32 %112, 0
  br i1 %113, label %114, label %117

114:                                              ; preds = %111
  %115 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %116 = load i32, i32 addrspace(2)* %115, align 4, !tbaa !47
  br label %204

117:                                              ; preds = %111
  %118 = add i32 %105, %19
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %119
  %121 = icmp sgt i32 %112, 0
  br i1 %121, label %125, label %122

122:                                              ; preds = %125, %117
  %123 = phi float [ 0.000000e+00, %117 ], [ %132, %125 ]
  %124 = icmp slt i32 %112, 4
  br i1 %124, label %136, label %142

125:                                              ; preds = %125, %117
  %126 = phi i32 [ %134, %125 ], [ 0, %117 ]
  %127 = phi float [ %132, %125 ], [ 0.000000e+00, %117 ]
  %128 = zext i32 %126 to i64
  %129 = getelementptr inbounds bfloat, bfloat addrspace(1)* %120, i64 %128
  %130 = load bfloat, bfloat addrspace(1)* %129, align 2, !tbaa !56
  %131 = fpext bfloat %130 to float
  %132 = fadd float %127, %131
  %133 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %128
  store float %131, float* %133, align 4, !tbaa !67
  %134 = add nuw nsw i32 %126, 1
  %135 = icmp eq i32 %134, %112
  br i1 %135, label %122, label %125, !llvm.loop !135

136:                                              ; preds = %136, %122
  %137 = phi i32 [ %140, %136 ], [ %112, %122 ]
  %138 = sext i32 %137 to i64
  %139 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %138
  store float 0.000000e+00, float* %139, align 4, !tbaa !67
  %140 = add i32 %137, 1
  %141 = icmp eq i32 %140, 4
  br i1 %141, label %142, label %136, !llvm.loop !136

142:                                              ; preds = %136, %122
  %143 = zext i32 %8 to i64
  %144 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %145 = load i64, i64 addrspace(2)* %144, align 8, !tbaa !54
  %146 = mul i64 %145, %143
  %147 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %148 = load i64, i64 addrspace(2)* %147, align 8, !tbaa !77
  %149 = mul i64 %148, %143
  %150 = lshr i32 %118, 6
  %151 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %152 = load i32, i32 addrspace(2)* %151, align 4, !tbaa !47
  %153 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %149
  %154 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %155 = load i64, i64 addrspace(2)* %154, align 8
  %156 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %157 = load i64, i64 addrspace(2)* %156, align 8
  %158 = zext i32 %150 to i64
  %159 = getelementptr inbounds i8, i8 addrspace(1)* %153, i64 %119
  br label %160

160:                                              ; preds = %201, %142
  %161 = phi i32 [ 0, %142 ], [ %202, %201 ]
  %162 = add i32 %161, %7
  %163 = icmp ult i32 %162, %152
  br i1 %163, label %164, label %201

164:                                              ; preds = %160
  %165 = zext i32 %162 to i64
  %166 = mul i64 %155, %165
  %167 = getelementptr inbounds i8, i8 addrspace(1)* %159, i64 %166
  %168 = mul i64 %157, %165
  %169 = add i64 %168, %146
  %170 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %169
  %171 = bitcast i8 addrspace(1)* %170 to bfloat addrspace(1)*
  %172 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %169
  %173 = bitcast i8 addrspace(1)* %172 to bfloat addrspace(1)*
  %174 = getelementptr inbounds bfloat, bfloat addrspace(1)* %171, i64 %158
  %175 = load bfloat, bfloat addrspace(1)* %174, align 2, !tbaa !56
  %176 = fpext bfloat %175 to float
  %177 = getelementptr inbounds bfloat, bfloat addrspace(1)* %173, i64 %158
  %178 = load bfloat, bfloat addrspace(1)* %177, align 2, !tbaa !56
  %179 = fpext bfloat %178 to float
  br i1 %121, label %180, label %193

180:                                              ; preds = %180, %164
  %181 = phi i32 [ %191, %180 ], [ 0, %164 ]
  %182 = phi float [ %190, %180 ], [ 0.000000e+00, %164 ]
  %183 = zext i32 %181 to i64
  %184 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %183
  %185 = load float, float* %184, align 4, !tbaa !67
  %186 = getelementptr inbounds i8, i8 addrspace(1)* %167, i64 %183
  %187 = load i8, i8 addrspace(1)* %186, align 1, !tbaa !78
  %188 = zext i8 %187 to i32
  %189 = tail call float @air.convert.f.f32.s.i32(i32 %188) #15
  %190 = tail call float @llvm.fmuladd.f32(float %185, float %189, float %182) #12
  %191 = add nuw nsw i32 %181, 1
  %192 = icmp eq i32 %191, %112
  br i1 %192, label %193, label %180, !llvm.loop !137

193:                                              ; preds = %180, %164
  %194 = phi float [ 0.000000e+00, %164 ], [ %190, %180 ]
  %195 = fmul float %123, %179
  %196 = tail call float @llvm.fmuladd.f32(float %176, float %194, float %195) #12
  %197 = zext i32 %161 to i64
  %198 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %197
  %199 = load float, float* %198, align 4, !tbaa !67
  %200 = fadd float %199, %196
  store float %200, float* %198, align 4, !tbaa !67
  br label %201

201:                                              ; preds = %193, %160
  %202 = add nuw nsw i32 %161, 1
  %203 = icmp eq i32 %202, 4
  br i1 %203, label %204, label %160, !llvm.loop !133

204:                                              ; preds = %201, %114
  %205 = phi i32 [ %116, %114 ], [ %152, %201 ]
  %206 = icmp eq i32 %10, 0
  %207 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %208 = zext i32 %205 to i64
  %209 = mul i64 %208, %9
  %210 = zext i32 %7 to i64
  %211 = add i64 %209, %210
  br label %213

212:                                              ; preds = %237
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %14) #12
  ret void

213:                                              ; preds = %237, %204
  %214 = phi i32 [ 0, %204 ], [ %238, %237 ]
  %215 = add i32 %214, %7
  %216 = icmp ult i32 %215, %205
  br i1 %216, label %217, label %237

217:                                              ; preds = %213
  %218 = zext i32 %214 to i64
  %219 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %218
  %220 = load float, float* %219, align 4, !tbaa !67
  %221 = tail call fast float @air.simd_sum.f32(float %220) #14
  br i1 %206, label %222, label %237

222:                                              ; preds = %217
  %223 = fptrunc float %221 to bfloat
  %224 = bitcast float %221 to i32
  %225 = and i32 %224, 2139095040
  %226 = icmp eq i32 %225, 2139095040
  br i1 %226, label %232, label %227

227:                                              ; preds = %222
  %228 = fpext bfloat %223 to float
  %229 = bitcast float %228 to i32
  %230 = and i32 %229, 2139095040
  %231 = icmp eq i32 %230, 2139095040
  br i1 %231, label %232, label %234

232:                                              ; preds = %227, %222
  %233 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %207, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %234

234:                                              ; preds = %232, %227
  %235 = add i64 %211, %218
  %236 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %235
  store bfloat %223, bfloat addrspace(1)* %236, align 2, !tbaa !56
  br label %237

237:                                              ; preds = %234, %217, %213
  %238 = add nuw nsw i32 %214, 1
  %239 = icmp eq i32 %238, 4
  br i1 %239, label %212, label %213, !llvm.loop !138
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %11
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !47
  br label %38

22:                                               ; preds = %11
  %23 = shl i32 %10, 3
  %24 = zext i32 %8 to i64
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %26 = load i64, i64 addrspace(2)* %25, align 8, !tbaa !54
  %27 = mul i64 %26, %24
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %29 = load i64, i64 addrspace(2)* %28, align 8, !tbaa !77
  %30 = mul i64 %29, %24
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %32 = load i32, i32 addrspace(2)* %31, align 4, !tbaa !47
  %33 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %30
  %34 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %35 = load i64, i64 addrspace(2)* %34, align 8
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %37 = load i64, i64 addrspace(2)* %36, align 8
  br label %46

38:                                               ; preds = %109, %19
  %39 = phi i32 [ %21, %19 ], [ %32, %109 ]
  %40 = icmp eq i32 %10, 0
  %41 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %42 = zext i32 %39 to i64
  %43 = mul i64 %42, %9
  %44 = zext i32 %7 to i64
  %45 = add i64 %43, %44
  br label %113

46:                                               ; preds = %109, %22
  %47 = phi i32 [ 0, %22 ], [ %110, %109 ]
  %48 = add i32 %47, %23
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %49
  br label %51

51:                                               ; preds = %51, %46
  %52 = phi i32 [ 0, %46 ], [ %60, %51 ]
  %53 = phi float [ 0.000000e+00, %46 ], [ %58, %51 ]
  %54 = zext i32 %52 to i64
  %55 = getelementptr inbounds bfloat, bfloat addrspace(1)* %50, i64 %54
  %56 = load bfloat, bfloat addrspace(1)* %55, align 2, !tbaa !56
  %57 = fpext bfloat %56 to float
  %58 = fadd float %53, %57
  %59 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %54
  store float %57, float* %59, align 4, !tbaa !67
  %60 = add nuw nsw i32 %52, 1
  %61 = icmp eq i32 %60, 8
  br i1 %61, label %62, label %51, !llvm.loop !126

62:                                               ; preds = %51
  %63 = lshr i32 %48, 7
  %64 = zext i32 %63 to i64
  %65 = getelementptr inbounds i8, i8 addrspace(1)* %33, i64 %49
  br label %66

66:                                               ; preds = %106, %62
  %67 = phi i32 [ 0, %62 ], [ %107, %106 ]
  %68 = add i32 %67, %7
  %69 = icmp ult i32 %68, %32
  br i1 %69, label %70, label %106

70:                                               ; preds = %66
  %71 = zext i32 %68 to i64
  %72 = mul i64 %35, %71
  %73 = getelementptr inbounds i8, i8 addrspace(1)* %65, i64 %72
  %74 = mul i64 %37, %71
  %75 = add i64 %74, %27
  %76 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %75
  %77 = bitcast i8 addrspace(1)* %76 to bfloat addrspace(1)*
  %78 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %75
  %79 = bitcast i8 addrspace(1)* %78 to bfloat addrspace(1)*
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %77, i64 %64
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !56
  %82 = getelementptr inbounds bfloat, bfloat addrspace(1)* %79, i64 %64
  %83 = load bfloat, bfloat addrspace(1)* %82, align 2, !tbaa !56
  br label %84

84:                                               ; preds = %84, %70
  %85 = phi i32 [ %95, %84 ], [ 0, %70 ]
  %86 = phi float [ %94, %84 ], [ 0.000000e+00, %70 ]
  %87 = zext i32 %85 to i64
  %88 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %87
  %89 = load float, float* %88, align 4, !tbaa !67
  %90 = getelementptr inbounds i8, i8 addrspace(1)* %73, i64 %87
  %91 = load i8, i8 addrspace(1)* %90, align 1, !tbaa !78
  %92 = zext i8 %91 to i32
  %93 = tail call float @air.convert.f.f32.s.i32(i32 %92) #15
  %94 = tail call float @llvm.fmuladd.f32(float %89, float %93, float %86) #12
  %95 = add nuw nsw i32 %85, 1
  %96 = icmp eq i32 %95, 8
  br i1 %96, label %97, label %84, !llvm.loop !127

97:                                               ; preds = %84
  %98 = fpext bfloat %81 to float
  %99 = fpext bfloat %83 to float
  %100 = fmul float %58, %99
  %101 = tail call float @llvm.fmuladd.f32(float %98, float %94, float %100) #12
  %102 = zext i32 %67 to i64
  %103 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %102
  %104 = load float, float* %103, align 4, !tbaa !67
  %105 = fadd float %104, %101
  store float %105, float* %103, align 4, !tbaa !67
  br label %106

106:                                              ; preds = %97, %66
  %107 = add nuw nsw i32 %67, 1
  %108 = icmp eq i32 %107, 4
  br i1 %108, label %109, label %66, !llvm.loop !139

109:                                              ; preds = %106
  %110 = add nuw i32 %47, 256
  %111 = icmp ult i32 %110, %17
  br i1 %111, label %46, label %38, !llvm.loop !140

112:                                              ; preds = %137
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #12
  ret void

113:                                              ; preds = %137, %38
  %114 = phi i32 [ 0, %38 ], [ %138, %137 ]
  %115 = add i32 %114, %7
  %116 = icmp ult i32 %115, %39
  br i1 %116, label %117, label %137

117:                                              ; preds = %113
  %118 = zext i32 %114 to i64
  %119 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %118
  %120 = load float, float* %119, align 4, !tbaa !67
  %121 = tail call fast float @air.simd_sum.f32(float %120) #14
  br i1 %40, label %122, label %137

122:                                              ; preds = %117
  %123 = fptrunc float %121 to bfloat
  %124 = bitcast float %121 to i32
  %125 = and i32 %124, 2139095040
  %126 = icmp eq i32 %125, 2139095040
  br i1 %126, label %132, label %127

127:                                              ; preds = %122
  %128 = fpext bfloat %123 to float
  %129 = bitcast float %128 to i32
  %130 = and i32 %129, 2139095040
  %131 = icmp eq i32 %130, 2139095040
  br i1 %131, label %132, label %134

132:                                              ; preds = %127, %122
  %133 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %41, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %134

134:                                              ; preds = %132, %127
  %135 = add i64 %45, %118
  %136 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %135
  store bfloat %123, bfloat addrspace(1)* %136, align 2, !tbaa !56
  br label %137

137:                                              ; preds = %134, %117, %113
  %138 = add nuw nsw i32 %114, 1
  %139 = icmp eq i32 %138, 4
  br i1 %139, label %112, label %113, !llvm.loop !141
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN25splash_mlx_qmv_f32xsum_v112project_mathILt8ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #3 {
  %12 = alloca [4 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [4 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %14) #12
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !48
  %18 = icmp ugt i32 %17, 128
  %19 = shl i32 %10, 2
  br i1 %18, label %20, label %104

20:                                               ; preds = %11
  %21 = zext i32 %8 to i64
  %22 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %23 = load i64, i64 addrspace(2)* %22, align 8, !tbaa !54
  %24 = mul i64 %23, %21
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %26 = load i64, i64 addrspace(2)* %25, align 8, !tbaa !77
  %27 = mul i64 %26, %21
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %29 = load i32, i32 addrspace(2)* %28, align 4, !tbaa !47
  %30 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %27
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %32 = load i64, i64 addrspace(2)* %31, align 8
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %34 = load i64, i64 addrspace(2)* %33, align 8
  br label %35

35:                                               ; preds = %98, %20
  %36 = phi i32 [ 0, %20 ], [ %99, %98 ]
  %37 = add i32 %36, %19
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %38
  br label %40

40:                                               ; preds = %40, %35
  %41 = phi i32 [ 0, %35 ], [ %49, %40 ]
  %42 = phi float [ 0.000000e+00, %35 ], [ %47, %40 ]
  %43 = zext i32 %41 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %39, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !56
  %46 = fpext bfloat %45 to float
  %47 = fadd float %42, %46
  %48 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %43
  store float %46, float* %48, align 4, !tbaa !67
  %49 = add nuw nsw i32 %41, 1
  %50 = icmp eq i32 %49, 4
  br i1 %50, label %51, label %40, !llvm.loop !131

51:                                               ; preds = %40
  %52 = lshr i32 %37, 7
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds i8, i8 addrspace(1)* %30, i64 %38
  br label %55

55:                                               ; preds = %95, %51
  %56 = phi i32 [ 0, %51 ], [ %96, %95 ]
  %57 = add i32 %56, %7
  %58 = icmp ult i32 %57, %29
  br i1 %58, label %59, label %95

59:                                               ; preds = %55
  %60 = zext i32 %57 to i64
  %61 = mul i64 %32, %60
  %62 = getelementptr inbounds i8, i8 addrspace(1)* %54, i64 %61
  %63 = mul i64 %34, %60
  %64 = add i64 %63, %24
  %65 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %64
  %66 = bitcast i8 addrspace(1)* %65 to bfloat addrspace(1)*
  %67 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %64
  %68 = bitcast i8 addrspace(1)* %67 to bfloat addrspace(1)*
  %69 = getelementptr inbounds bfloat, bfloat addrspace(1)* %66, i64 %53
  %70 = load bfloat, bfloat addrspace(1)* %69, align 2, !tbaa !56
  %71 = getelementptr inbounds bfloat, bfloat addrspace(1)* %68, i64 %53
  %72 = load bfloat, bfloat addrspace(1)* %71, align 2, !tbaa !56
  br label %73

73:                                               ; preds = %73, %59
  %74 = phi i32 [ %84, %73 ], [ 0, %59 ]
  %75 = phi float [ %83, %73 ], [ 0.000000e+00, %59 ]
  %76 = zext i32 %74 to i64
  %77 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %76
  %78 = load float, float* %77, align 4, !tbaa !67
  %79 = getelementptr inbounds i8, i8 addrspace(1)* %62, i64 %76
  %80 = load i8, i8 addrspace(1)* %79, align 1, !tbaa !78
  %81 = zext i8 %80 to i32
  %82 = tail call float @air.convert.f.f32.s.i32(i32 %81) #15
  %83 = tail call float @llvm.fmuladd.f32(float %78, float %82, float %75) #12
  %84 = add nuw nsw i32 %74, 1
  %85 = icmp eq i32 %84, 4
  br i1 %85, label %86, label %73, !llvm.loop !132

86:                                               ; preds = %73
  %87 = fpext bfloat %70 to float
  %88 = fpext bfloat %72 to float
  %89 = fmul float %47, %88
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %83, float %89) #12
  %91 = zext i32 %56 to i64
  %92 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %91
  %93 = load float, float* %92, align 4, !tbaa !67
  %94 = fadd float %93, %90
  store float %94, float* %92, align 4, !tbaa !67
  br label %95

95:                                               ; preds = %86, %55
  %96 = add nuw nsw i32 %56, 1
  %97 = icmp eq i32 %96, 4
  br i1 %97, label %98, label %55, !llvm.loop !142

98:                                               ; preds = %95
  %99 = add i32 %36, 128
  %100 = icmp ugt i32 %17, %99
  %101 = sub i32 %17, %99
  %102 = icmp ugt i32 %101, 128
  %103 = and i1 %100, %102
  br i1 %103, label %35, label %104, !llvm.loop !143

104:                                              ; preds = %98, %11
  %105 = phi i32 [ 0, %11 ], [ %99, %98 ]
  %106 = phi i32 [ %17, %11 ], [ %101, %98 ]
  %107 = icmp ugt i32 %106, %19
  br i1 %107, label %108, label %111

108:                                              ; preds = %104
  %109 = sub i32 %106, %19
  %110 = tail call i32 @air.min.u.i32(i32 %109, i32 4) #15
  br label %111

111:                                              ; preds = %108, %104
  %112 = phi i32 [ %110, %108 ], [ 0, %104 ]
  %113 = icmp eq i32 %112, 0
  br i1 %113, label %114, label %117

114:                                              ; preds = %111
  %115 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %116 = load i32, i32 addrspace(2)* %115, align 4, !tbaa !47
  br label %204

117:                                              ; preds = %111
  %118 = add i32 %105, %19
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %119
  %121 = icmp sgt i32 %112, 0
  br i1 %121, label %125, label %122

122:                                              ; preds = %125, %117
  %123 = phi float [ 0.000000e+00, %117 ], [ %132, %125 ]
  %124 = icmp slt i32 %112, 4
  br i1 %124, label %136, label %142

125:                                              ; preds = %125, %117
  %126 = phi i32 [ %134, %125 ], [ 0, %117 ]
  %127 = phi float [ %132, %125 ], [ 0.000000e+00, %117 ]
  %128 = zext i32 %126 to i64
  %129 = getelementptr inbounds bfloat, bfloat addrspace(1)* %120, i64 %128
  %130 = load bfloat, bfloat addrspace(1)* %129, align 2, !tbaa !56
  %131 = fpext bfloat %130 to float
  %132 = fadd float %127, %131
  %133 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %128
  store float %131, float* %133, align 4, !tbaa !67
  %134 = add nuw nsw i32 %126, 1
  %135 = icmp eq i32 %134, %112
  br i1 %135, label %122, label %125, !llvm.loop !135

136:                                              ; preds = %136, %122
  %137 = phi i32 [ %140, %136 ], [ %112, %122 ]
  %138 = sext i32 %137 to i64
  %139 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %138
  store float 0.000000e+00, float* %139, align 4, !tbaa !67
  %140 = add i32 %137, 1
  %141 = icmp eq i32 %140, 4
  br i1 %141, label %142, label %136, !llvm.loop !136

142:                                              ; preds = %136, %122
  %143 = zext i32 %8 to i64
  %144 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %145 = load i64, i64 addrspace(2)* %144, align 8, !tbaa !54
  %146 = mul i64 %145, %143
  %147 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %148 = load i64, i64 addrspace(2)* %147, align 8, !tbaa !77
  %149 = mul i64 %148, %143
  %150 = lshr i32 %118, 7
  %151 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %152 = load i32, i32 addrspace(2)* %151, align 4, !tbaa !47
  %153 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %149
  %154 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %155 = load i64, i64 addrspace(2)* %154, align 8
  %156 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %157 = load i64, i64 addrspace(2)* %156, align 8
  %158 = zext i32 %150 to i64
  %159 = getelementptr inbounds i8, i8 addrspace(1)* %153, i64 %119
  br label %160

160:                                              ; preds = %201, %142
  %161 = phi i32 [ 0, %142 ], [ %202, %201 ]
  %162 = add i32 %161, %7
  %163 = icmp ult i32 %162, %152
  br i1 %163, label %164, label %201

164:                                              ; preds = %160
  %165 = zext i32 %162 to i64
  %166 = mul i64 %155, %165
  %167 = getelementptr inbounds i8, i8 addrspace(1)* %159, i64 %166
  %168 = mul i64 %157, %165
  %169 = add i64 %168, %146
  %170 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %169
  %171 = bitcast i8 addrspace(1)* %170 to bfloat addrspace(1)*
  %172 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %169
  %173 = bitcast i8 addrspace(1)* %172 to bfloat addrspace(1)*
  %174 = getelementptr inbounds bfloat, bfloat addrspace(1)* %171, i64 %158
  %175 = load bfloat, bfloat addrspace(1)* %174, align 2, !tbaa !56
  %176 = fpext bfloat %175 to float
  %177 = getelementptr inbounds bfloat, bfloat addrspace(1)* %173, i64 %158
  %178 = load bfloat, bfloat addrspace(1)* %177, align 2, !tbaa !56
  %179 = fpext bfloat %178 to float
  br i1 %121, label %180, label %193

180:                                              ; preds = %180, %164
  %181 = phi i32 [ %191, %180 ], [ 0, %164 ]
  %182 = phi float [ %190, %180 ], [ 0.000000e+00, %164 ]
  %183 = zext i32 %181 to i64
  %184 = getelementptr inbounds [4 x float], [4 x float]* %12, i64 0, i64 %183
  %185 = load float, float* %184, align 4, !tbaa !67
  %186 = getelementptr inbounds i8, i8 addrspace(1)* %167, i64 %183
  %187 = load i8, i8 addrspace(1)* %186, align 1, !tbaa !78
  %188 = zext i8 %187 to i32
  %189 = tail call float @air.convert.f.f32.s.i32(i32 %188) #15
  %190 = tail call float @llvm.fmuladd.f32(float %185, float %189, float %182) #12
  %191 = add nuw nsw i32 %181, 1
  %192 = icmp eq i32 %191, %112
  br i1 %192, label %193, label %180, !llvm.loop !137

193:                                              ; preds = %180, %164
  %194 = phi float [ 0.000000e+00, %164 ], [ %190, %180 ]
  %195 = fmul float %123, %179
  %196 = tail call float @llvm.fmuladd.f32(float %176, float %194, float %195) #12
  %197 = zext i32 %161 to i64
  %198 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %197
  %199 = load float, float* %198, align 4, !tbaa !67
  %200 = fadd float %199, %196
  store float %200, float* %198, align 4, !tbaa !67
  br label %201

201:                                              ; preds = %193, %160
  %202 = add nuw nsw i32 %161, 1
  %203 = icmp eq i32 %202, 4
  br i1 %203, label %204, label %160, !llvm.loop !142

204:                                              ; preds = %201, %114
  %205 = phi i32 [ %116, %114 ], [ %152, %201 ]
  %206 = icmp eq i32 %10, 0
  %207 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %208 = zext i32 %205 to i64
  %209 = mul i64 %208, %9
  %210 = zext i32 %7 to i64
  %211 = add i64 %209, %210
  br label %213

212:                                              ; preds = %237
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #12
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %14) #12
  ret void

213:                                              ; preds = %237, %204
  %214 = phi i32 [ 0, %204 ], [ %238, %237 ]
  %215 = add i32 %214, %7
  %216 = icmp ult i32 %215, %205
  br i1 %216, label %217, label %237

217:                                              ; preds = %213
  %218 = zext i32 %214 to i64
  %219 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %218
  %220 = load float, float* %219, align 4, !tbaa !67
  %221 = tail call fast float @air.simd_sum.f32(float %220) #14
  br i1 %206, label %222, label %237

222:                                              ; preds = %217
  %223 = fptrunc float %221 to bfloat
  %224 = bitcast float %221 to i32
  %225 = and i32 %224, 2139095040
  %226 = icmp eq i32 %225, 2139095040
  br i1 %226, label %232, label %227

227:                                              ; preds = %222
  %228 = fpext bfloat %223 to float
  %229 = bitcast float %228 to i32
  %230 = and i32 %229, 2139095040
  %231 = icmp eq i32 %230, 2139095040
  br i1 %231, label %232, label %234

232:                                              ; preds = %227, %222
  %233 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %207, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %234

234:                                              ; preds = %232, %227
  %235 = add i64 %211, %218
  %236 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %235
  store bfloat %223, bfloat addrspace(1)* %236, align 2, !tbaa !56
  br label %237

237:                                              ; preds = %234, %217, %213
  %238 = add nuw nsw i32 %214, 1
  %239 = icmp eq i32 %238, 4
  br i1 %239, label %212, label %213, !llvm.loop !144
}

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #2 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #3 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #4 = { mustprogress nounwind willreturn }
attributes #5 = { argmemonly nofree nounwind willreturn writeonly }
attributes #6 = { inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #7 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #8 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #9 = { convergent mustprogress nounwind willreturn }
attributes #10 = { convergent nobuiltin "no-builtins" }
attributes #11 = { nounwind willreturn }
attributes #12 = { nounwind }
attributes #13 = { nobuiltin "no-builtins" }
attributes #14 = { convergent nounwind willreturn }
attributes #15 = { nounwind readnone willreturn }
attributes #16 = { nobuiltin nounwind "no-builtins" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !25, !26, !27, !28, !29, !30, !31}
!air.compile_options = !{!32, !33, !34}
!llvm.ident = !{!35}
!air.version = !{!36}
!air.language_version = !{!37}
!air.source_file_name = !{!38}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q4_g64, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !20, !22, !23, !24}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"input"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"weights"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"scales"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"biases"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"long", !"air.arg_name", !"expert_ids"}
!17 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"output"}
!18 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !19, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"diagnostics"}
!19 = !{i32 0, i32 4, i32 0, !"uint", !"__s"}
!20 = !{i32 7, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !21, !"air.arg_type_size", i32 64, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"FlashAffineParams", !"air.arg_name", !"p"}
!21 = !{i32 0, i32 4, i32 0, !"uint", !"rows", i32 4, i32 4, i32 0, !"uint", !"selections", i32 8, i32 4, i32 0, !"uint", !"input_size", i32 12, i32 4, i32 0, !"uint", !"output_size", i32 16, i32 4, i32 0, !"uint", !"experts", i32 20, i32 4, i32 0, !"uint", !"bits", i32 24, i32 4, i32 0, !"uint", !"group_size", i32 28, i32 4, i32 0, !"uint", !"flags", i32 32, i32 8, i32 0, !"ulong", !"weight_row_stride_bytes", i32 40, i32 8, i32 0, !"ulong", !"weight_expert_stride_bytes", i32 48, i32 8, i32 0, !"ulong", !"parameter_row_stride_bytes", i32 56, i32 8, i32 0, !"ulong", !"parameter_expert_stride_bytes"}
!22 = !{i32 8, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"group"}
!23 = !{i32 9, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
!24 = !{i32 10, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!25 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q4_g128, !10, !11}
!26 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q5_g64, !10, !11}
!27 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q5_g128, !10, !11}
!28 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q6_g64, !10, !11}
!29 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q6_g128, !10, !11}
!30 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q8_g64, !10, !11}
!31 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, i32, i32)* @flash_affine_mlx_qmv_f32xsum_v1_q8_g128, !10, !11}
!32 = !{!"air.compile.denorms_disable"}
!33 = !{!"air.compile.fast_math_enable"}
!34 = !{!"air.compile.framebuffer_fetch_enable"}
!35 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!36 = !{i32 2, i32 9, i32 0}
!37 = !{!"Metal", i32 4, i32 1, i32 0}
!38 = !{!"/Users/mweinbach/Projects/splash/runtime/metal/kernels/shared/flash_affine_qmv_f32.metal"}
!39 = !{!40, !41, i64 0}
!40 = !{!"_ZTS17FlashAffineParams", !41, i64 0, !41, i64 4, !41, i64 8, !41, i64 12, !41, i64 16, !41, i64 20, !41, i64 24, !41, i64 28, !44, i64 32, !44, i64 40, !44, i64 48, !44, i64 56}
!41 = !{!"int", !42, i64 0}
!42 = !{!"omnipotent char", !43, i64 0}
!43 = !{!"Simple C++ TBAA"}
!44 = !{!"long", !42, i64 0}
!45 = !{!40, !41, i64 4}
!46 = !{!40, !41, i64 16}
!47 = !{!40, !41, i64 12}
!48 = !{!40, !41, i64 8}
!49 = !{!40, !41, i64 20}
!50 = !{!40, !41, i64 24}
!51 = !{!40, !41, i64 28}
!52 = !{!40, !44, i64 32}
!53 = !{!40, !44, i64 48}
!54 = !{!40, !44, i64 56}
!55 = !{!44, !44, i64 0}
!56 = !{!57, !57, i64 0}
!57 = !{!"bfloat", !42, i64 0}
!58 = distinct !{!58, !59}
!59 = !{!"llvm.loop.mustprogress"}
!60 = distinct !{!60, !59}
!61 = distinct !{!61, !59}
!62 = distinct !{!62, !59}
!63 = distinct !{!63, !59}
!64 = distinct !{!64, !59}
!65 = distinct !{!65, !59}
!66 = distinct !{!66, !59}
!67 = !{!68, !68, i64 0}
!68 = !{!"float", !42, i64 0}
!69 = distinct !{!69, !59}
!70 = distinct !{!70, !59}
!71 = distinct !{!71, !59}
!72 = distinct !{!72, !59}
!73 = distinct !{!73, !59}
!74 = distinct !{!74, !59}
!75 = distinct !{!75, !59}
!76 = distinct !{!76, !59}
!77 = !{!40, !44, i64 40}
!78 = !{!42, !42, i64 0}
!79 = distinct !{!79, !59}
!80 = distinct !{!80, !59}
!81 = distinct !{!81, !59}
!82 = distinct !{!82, !59}
!83 = distinct !{!83, !59}
!84 = distinct !{!84, !59}
!85 = distinct !{!85, !59}
!86 = distinct !{!86, !59}
!87 = distinct !{!87, !59}
!88 = distinct !{!88, !59}
!89 = distinct !{!89, !59}
!90 = distinct !{!90, !59}
!91 = distinct !{!91, !59}
!92 = distinct !{!92, !59}
!93 = distinct !{!93, !59}
!94 = distinct !{!94, !59}
!95 = distinct !{!95, !59}
!96 = distinct !{!96, !59}
!97 = distinct !{!97, !59}
!98 = distinct !{!98, !59}
!99 = distinct !{!99, !59}
!100 = distinct !{!100, !59}
!101 = distinct !{!101, !59}
!102 = distinct !{!102, !59}
!103 = distinct !{!103, !59}
!104 = distinct !{!104, !59}
!105 = distinct !{!105, !59}
!106 = distinct !{!106, !59}
!107 = distinct !{!107, !59}
!108 = distinct !{!108, !59}
!109 = distinct !{!109, !59}
!110 = distinct !{!110, !59}
!111 = distinct !{!111, !59}
!112 = distinct !{!112, !59}
!113 = distinct !{!113, !59}
!114 = distinct !{!114, !59}
!115 = distinct !{!115, !59}
!116 = distinct !{!116, !59}
!117 = distinct !{!117, !59}
!118 = distinct !{!118, !59}
!119 = distinct !{!119, !59}
!120 = distinct !{!120, !59}
!121 = distinct !{!121, !59}
!122 = distinct !{!122, !59}
!123 = distinct !{!123, !59}
!124 = distinct !{!124, !59}
!125 = distinct !{!125, !59}
!126 = distinct !{!126, !59}
!127 = distinct !{!127, !59}
!128 = distinct !{!128, !59}
!129 = distinct !{!129, !59}
!130 = distinct !{!130, !59}
!131 = distinct !{!131, !59}
!132 = distinct !{!132, !59}
!133 = distinct !{!133, !59}
!134 = distinct !{!134, !59}
!135 = distinct !{!135, !59}
!136 = distinct !{!136, !59}
!137 = distinct !{!137, !59}
!138 = distinct !{!138, !59}
!139 = distinct !{!139, !59}
!140 = distinct !{!140, !59}
!141 = distinct !{!141, !59}
!142 = distinct !{!142, !59}
!143 = distinct !{!143, !59}
!144 = distinct !{!144, !59}
