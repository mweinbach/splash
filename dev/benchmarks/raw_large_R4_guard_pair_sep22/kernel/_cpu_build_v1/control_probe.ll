; ModuleID = '/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/_cpu_build_v1/control_probe.air'
source_filename = "/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/control_probe.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

%"struct.metal::_atomic" = type { i32 }
%struct.FlashAffineParams = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64 }

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_control_probe_q4_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #10
  %17 = icmp ne i32 %11, 32
  %18 = or i1 %17, %16
  %19 = icmp ugt i32 %12, 1
  %20 = or i1 %19, %18
  %21 = icmp ugt i32 %13, 31
  %22 = or i1 %21, %20
  br i1 %22, label %23, label %28

23:                                               ; preds = %14
  %24 = icmp eq i32 %13, 0
  br i1 %24, label %25, label %29

25:                                               ; preds = %23
  %26 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN22r5_raw_odd_control_tap7projectILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap7projectILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #1 {
  %13 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 0
  %14 = load i32, i32 addrspace(2)* %13, align 8, !tbaa !38
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %78, label %16

16:                                               ; preds = %12
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 1
  %18 = load i32, i32 addrspace(2)* %17, align 4, !tbaa !44
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %78, label %20

20:                                               ; preds = %16
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 4
  %22 = load i32, i32 addrspace(2)* %21, align 8, !tbaa !45
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %78, label %24

24:                                               ; preds = %20
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !46
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %78, label %28

28:                                               ; preds = %24
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 2
  %30 = load i32, i32 addrspace(2)* %29, align 8, !tbaa !47
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %78, label %32

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 5
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !48
  %35 = icmp eq i32 %34, 4
  br i1 %35, label %36, label %78

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 6
  %38 = load i32, i32 addrspace(2)* %37, align 8, !tbaa !49
  %39 = icmp eq i32 %38, 64
  %40 = and i32 %30, 63
  %41 = icmp eq i32 %40, 0
  %42 = select i1 %39, i1 %41, i1 false
  br i1 %42, label %43, label %78

43:                                               ; preds = %36
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 7
  %45 = load i32, i32 addrspace(2)* %44, align 4, !tbaa !50
  %46 = icmp ult i32 %45, 4
  br i1 %46, label %47, label %78

47:                                               ; preds = %43
  %48 = and i32 %45, 2
  %49 = and i32 %45, 1
  %50 = icmp eq i32 %49, 0
  %51 = icmp eq i32 %45, 2
  br i1 %51, label %78, label %52

52:                                               ; preds = %47
  br i1 %50, label %53, label %57

53:                                               ; preds = %52
  %54 = icmp eq i32 %22, 1
  %55 = icmp eq i32 %18, 1
  %56 = select i1 %54, i1 %55, i1 false
  br i1 %56, label %57, label %78

57:                                               ; preds = %53, %52
  %58 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 8
  %59 = load i64, i64 addrspace(2)* %58, align 8, !tbaa !51
  %60 = lshr i32 %30, 1
  %61 = zext i32 %60 to i64
  %62 = icmp ult i64 %59, %61
  br i1 %62, label %78, label %63

63:                                               ; preds = %57
  %64 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 10
  %65 = load i64, i64 addrspace(2)* %64, align 8, !tbaa !52
  %66 = lshr i32 %30, 5
  %67 = and i32 %66, 134217726
  %68 = zext i32 %67 to i64
  %69 = icmp uge i64 %65, %68
  %70 = and i64 %65, 1
  %71 = icmp eq i64 %70, 0
  %72 = and i1 %69, %71
  br i1 %72, label %73, label %78

73:                                               ; preds = %63
  %74 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 11
  %75 = load i64, i64 addrspace(2)* %74, align 8, !tbaa !53
  %76 = and i64 %75, 1
  %77 = icmp eq i64 %76, 0
  br i1 %77, label %83, label %78

78:                                               ; preds = %73, %63, %57, %53, %47, %43, %36, %32, %28, %24, %20, %16, %12
  %79 = icmp eq i32 %11, 0
  br i1 %79, label %80, label %143

80:                                               ; preds = %78
  %81 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %82 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %81, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %143

83:                                               ; preds = %73
  %84 = extractelement <3 x i32> %9, i64 0
  %85 = shl i32 %84, 3
  %86 = shl i32 %10, 2
  %87 = add i32 %85, %86
  %88 = icmp ult i32 %87, %26
  br i1 %88, label %89, label %143

89:                                               ; preds = %83
  %90 = extractelement <3 x i32> %9, i64 1
  %91 = icmp ult i32 %90, %14
  br i1 %91, label %92, label %143

92:                                               ; preds = %89
  %93 = extractelement <3 x i32> %9, i64 2
  %94 = icmp ult i32 %93, %18
  br i1 %94, label %95, label %143

95:                                               ; preds = %92
  %96 = zext i32 %90 to i64
  %97 = zext i32 %18 to i64
  %98 = mul nuw i64 %97, %96
  %99 = zext i32 %93 to i64
  %100 = add nuw i64 %98, %99
  br i1 %50, label %104, label %101

101:                                              ; preds = %95
  %102 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %100
  %103 = load i64, i64 addrspace(1)* %102, align 8, !tbaa !54
  br label %104

104:                                              ; preds = %101, %95
  %105 = phi i64 [ %103, %101 ], [ 0, %95 ]
  %106 = icmp sgt i64 %105, -1
  %107 = zext i32 %22 to i64
  %108 = icmp ult i64 %105, %107
  %109 = select i1 %106, i1 %108, i1 false
  br i1 %109, label %130, label %110

110:                                              ; preds = %104
  %111 = icmp eq i32 %11, 0
  br i1 %111, label %112, label %143

112:                                              ; preds = %110
  %113 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %114 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %113, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %115 = zext i32 %26 to i64
  %116 = mul i64 %100, %115
  %117 = zext i32 %87 to i64
  %118 = add i64 %116, %117
  br label %119

119:                                              ; preds = %127, %112
  %120 = phi i32 [ 0, %112 ], [ %128, %127 ]
  %121 = add nuw nsw i32 %120, %87
  %122 = icmp ult i32 %121, %26
  br i1 %122, label %123, label %127

123:                                              ; preds = %119
  %124 = zext i32 %120 to i64
  %125 = add i64 %118, %124
  %126 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %125
  store bfloat 0xR7FC0, bfloat addrspace(1)* %126, align 2, !tbaa !55
  br label %127

127:                                              ; preds = %123, %119
  %128 = add nuw nsw i32 %120, 1
  %129 = icmp eq i32 %128, 4
  br i1 %129, label %143, label %119, !llvm.loop !57

130:                                              ; preds = %104
  %131 = icmp eq i32 %48, 0
  %132 = select i1 %131, i64 %96, i64 %100
  %133 = zext i32 %30 to i64
  %134 = mul i64 %132, %133
  %135 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %134
  %136 = icmp ugt i32 %26, 7
  %137 = and i32 %30, 511
  %138 = icmp eq i32 %137, 0
  %139 = select i1 %136, i1 %138, i1 false
  %140 = trunc i64 %105 to i32
  br i1 %139, label %141, label %142

141:                                              ; preds = %130
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %87, i32 noundef %140, i64 noundef %100, i32 noundef %11) #12
  br label %143

142:                                              ; preds = %130
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt4ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %87, i32 noundef %140, i64 noundef %100, i32 noundef %11) #12
  br label %143

143:                                              ; preds = %142, %141, %127, %110, %92, %89, %83, %80, %78
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_control_probe_q5_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #10
  %17 = icmp ne i32 %11, 32
  %18 = or i1 %17, %16
  %19 = icmp ugt i32 %12, 1
  %20 = or i1 %19, %18
  %21 = icmp ugt i32 %13, 31
  %22 = or i1 %21, %20
  br i1 %22, label %23, label %28

23:                                               ; preds = %14
  %24 = icmp eq i32 %13, 0
  br i1 %24, label %25, label %29

25:                                               ; preds = %23
  %26 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN22r5_raw_odd_control_tap7projectILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap7projectILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #1 {
  %13 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 0
  %14 = load i32, i32 addrspace(2)* %13, align 8, !tbaa !38
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %79, label %16

16:                                               ; preds = %12
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 1
  %18 = load i32, i32 addrspace(2)* %17, align 4, !tbaa !44
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %79, label %20

20:                                               ; preds = %16
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 4
  %22 = load i32, i32 addrspace(2)* %21, align 8, !tbaa !45
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %79, label %24

24:                                               ; preds = %20
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !46
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %79, label %28

28:                                               ; preds = %24
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 2
  %30 = load i32, i32 addrspace(2)* %29, align 8, !tbaa !47
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %79, label %32

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 5
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !48
  %35 = icmp eq i32 %34, 5
  br i1 %35, label %36, label %79

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 6
  %38 = load i32, i32 addrspace(2)* %37, align 8, !tbaa !49
  %39 = icmp eq i32 %38, 64
  %40 = and i32 %30, 63
  %41 = icmp eq i32 %40, 0
  %42 = select i1 %39, i1 %41, i1 false
  br i1 %42, label %43, label %79

43:                                               ; preds = %36
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 7
  %45 = load i32, i32 addrspace(2)* %44, align 4, !tbaa !50
  %46 = icmp ult i32 %45, 4
  br i1 %46, label %47, label %79

47:                                               ; preds = %43
  %48 = and i32 %45, 2
  %49 = and i32 %45, 1
  %50 = icmp eq i32 %49, 0
  %51 = icmp eq i32 %45, 2
  br i1 %51, label %79, label %52

52:                                               ; preds = %47
  br i1 %50, label %53, label %57

53:                                               ; preds = %52
  %54 = icmp eq i32 %22, 1
  %55 = icmp eq i32 %18, 1
  %56 = select i1 %54, i1 %55, i1 false
  br i1 %56, label %57, label %79

57:                                               ; preds = %53, %52
  %58 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 8
  %59 = load i64, i64 addrspace(2)* %58, align 8, !tbaa !51
  %60 = zext i32 %30 to i64
  %61 = mul nuw nsw i64 %60, 5
  %62 = lshr i64 %61, 3
  %63 = icmp ult i64 %59, %62
  br i1 %63, label %79, label %64

64:                                               ; preds = %57
  %65 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 10
  %66 = load i64, i64 addrspace(2)* %65, align 8, !tbaa !52
  %67 = lshr i32 %30, 5
  %68 = and i32 %67, 134217726
  %69 = zext i32 %68 to i64
  %70 = icmp uge i64 %66, %69
  %71 = and i64 %66, 1
  %72 = icmp eq i64 %71, 0
  %73 = and i1 %70, %72
  br i1 %73, label %74, label %79

74:                                               ; preds = %64
  %75 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 11
  %76 = load i64, i64 addrspace(2)* %75, align 8, !tbaa !53
  %77 = and i64 %76, 1
  %78 = icmp eq i64 %77, 0
  br i1 %78, label %84, label %79

79:                                               ; preds = %74, %64, %57, %53, %47, %43, %36, %32, %28, %24, %20, %16, %12
  %80 = icmp eq i32 %11, 0
  br i1 %80, label %81, label %143

81:                                               ; preds = %79
  %82 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %83 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %82, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %143

84:                                               ; preds = %74
  %85 = extractelement <3 x i32> %9, i64 0
  %86 = shl i32 %85, 3
  %87 = shl i32 %10, 2
  %88 = add i32 %86, %87
  %89 = icmp ult i32 %88, %26
  br i1 %89, label %90, label %143

90:                                               ; preds = %84
  %91 = extractelement <3 x i32> %9, i64 1
  %92 = icmp ult i32 %91, %14
  br i1 %92, label %93, label %143

93:                                               ; preds = %90
  %94 = extractelement <3 x i32> %9, i64 2
  %95 = icmp ult i32 %94, %18
  br i1 %95, label %96, label %143

96:                                               ; preds = %93
  %97 = zext i32 %91 to i64
  %98 = zext i32 %18 to i64
  %99 = mul nuw i64 %98, %97
  %100 = zext i32 %94 to i64
  %101 = add nuw i64 %99, %100
  br i1 %50, label %105, label %102

102:                                              ; preds = %96
  %103 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %101
  %104 = load i64, i64 addrspace(1)* %103, align 8, !tbaa !54
  br label %105

105:                                              ; preds = %102, %96
  %106 = phi i64 [ %104, %102 ], [ 0, %96 ]
  %107 = icmp sgt i64 %106, -1
  %108 = zext i32 %22 to i64
  %109 = icmp ult i64 %106, %108
  %110 = select i1 %107, i1 %109, i1 false
  br i1 %110, label %131, label %111

111:                                              ; preds = %105
  %112 = icmp eq i32 %11, 0
  br i1 %112, label %113, label %143

113:                                              ; preds = %111
  %114 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %115 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %114, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %116 = zext i32 %26 to i64
  %117 = mul i64 %101, %116
  %118 = zext i32 %88 to i64
  %119 = add i64 %117, %118
  br label %120

120:                                              ; preds = %128, %113
  %121 = phi i32 [ 0, %113 ], [ %129, %128 ]
  %122 = add nuw nsw i32 %121, %88
  %123 = icmp ult i32 %122, %26
  br i1 %123, label %124, label %128

124:                                              ; preds = %120
  %125 = zext i32 %121 to i64
  %126 = add i64 %119, %125
  %127 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %126
  store bfloat 0xR7FC0, bfloat addrspace(1)* %127, align 2, !tbaa !55
  br label %128

128:                                              ; preds = %124, %120
  %129 = add nuw nsw i32 %121, 1
  %130 = icmp eq i32 %129, 4
  br i1 %130, label %143, label %120, !llvm.loop !59

131:                                              ; preds = %105
  %132 = icmp eq i32 %48, 0
  %133 = select i1 %132, i64 %97, i64 %101
  %134 = mul i64 %133, %60
  %135 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %134
  %136 = icmp ugt i32 %26, 7
  %137 = and i32 %30, 511
  %138 = icmp eq i32 %137, 0
  %139 = select i1 %136, i1 %138, i1 false
  %140 = trunc i64 %106 to i32
  br i1 %139, label %141, label %142

141:                                              ; preds = %131
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %88, i32 noundef %140, i64 noundef %101, i32 noundef %11) #12
  br label %143

142:                                              ; preds = %131
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %88, i32 noundef %140, i64 noundef %101, i32 noundef %11) #12
  br label %143

143:                                              ; preds = %142, %141, %128, %111, %93, %90, %84, %81, %79
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_control_probe_q5_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #10
  %17 = icmp ne i32 %11, 32
  %18 = or i1 %17, %16
  %19 = icmp ugt i32 %12, 1
  %20 = or i1 %19, %18
  %21 = icmp ugt i32 %13, 31
  %22 = or i1 %21, %20
  br i1 %22, label %23, label %28

23:                                               ; preds = %14
  %24 = icmp eq i32 %13, 0
  br i1 %24, label %25, label %29

25:                                               ; preds = %23
  %26 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN22r5_raw_odd_control_tap7projectILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap7projectILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #1 {
  %13 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 0
  %14 = load i32, i32 addrspace(2)* %13, align 8, !tbaa !38
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %79, label %16

16:                                               ; preds = %12
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 1
  %18 = load i32, i32 addrspace(2)* %17, align 4, !tbaa !44
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %79, label %20

20:                                               ; preds = %16
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 4
  %22 = load i32, i32 addrspace(2)* %21, align 8, !tbaa !45
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %79, label %24

24:                                               ; preds = %20
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !46
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %79, label %28

28:                                               ; preds = %24
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 2
  %30 = load i32, i32 addrspace(2)* %29, align 8, !tbaa !47
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %79, label %32

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 5
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !48
  %35 = icmp eq i32 %34, 5
  br i1 %35, label %36, label %79

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 6
  %38 = load i32, i32 addrspace(2)* %37, align 8, !tbaa !49
  %39 = icmp eq i32 %38, 128
  %40 = and i32 %30, 127
  %41 = icmp eq i32 %40, 0
  %42 = select i1 %39, i1 %41, i1 false
  br i1 %42, label %43, label %79

43:                                               ; preds = %36
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 7
  %45 = load i32, i32 addrspace(2)* %44, align 4, !tbaa !50
  %46 = icmp ult i32 %45, 4
  br i1 %46, label %47, label %79

47:                                               ; preds = %43
  %48 = and i32 %45, 2
  %49 = and i32 %45, 1
  %50 = icmp eq i32 %49, 0
  %51 = icmp eq i32 %45, 2
  br i1 %51, label %79, label %52

52:                                               ; preds = %47
  br i1 %50, label %53, label %57

53:                                               ; preds = %52
  %54 = icmp eq i32 %22, 1
  %55 = icmp eq i32 %18, 1
  %56 = select i1 %54, i1 %55, i1 false
  br i1 %56, label %57, label %79

57:                                               ; preds = %53, %52
  %58 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 8
  %59 = load i64, i64 addrspace(2)* %58, align 8, !tbaa !51
  %60 = zext i32 %30 to i64
  %61 = mul nuw nsw i64 %60, 5
  %62 = lshr i64 %61, 3
  %63 = icmp ult i64 %59, %62
  br i1 %63, label %79, label %64

64:                                               ; preds = %57
  %65 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 10
  %66 = load i64, i64 addrspace(2)* %65, align 8, !tbaa !52
  %67 = lshr i32 %30, 6
  %68 = and i32 %67, 67108862
  %69 = zext i32 %68 to i64
  %70 = icmp uge i64 %66, %69
  %71 = and i64 %66, 1
  %72 = icmp eq i64 %71, 0
  %73 = and i1 %70, %72
  br i1 %73, label %74, label %79

74:                                               ; preds = %64
  %75 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 11
  %76 = load i64, i64 addrspace(2)* %75, align 8, !tbaa !53
  %77 = and i64 %76, 1
  %78 = icmp eq i64 %77, 0
  br i1 %78, label %84, label %79

79:                                               ; preds = %74, %64, %57, %53, %47, %43, %36, %32, %28, %24, %20, %16, %12
  %80 = icmp eq i32 %11, 0
  br i1 %80, label %81, label %143

81:                                               ; preds = %79
  %82 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %83 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %82, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %143

84:                                               ; preds = %74
  %85 = extractelement <3 x i32> %9, i64 0
  %86 = shl i32 %85, 3
  %87 = shl i32 %10, 2
  %88 = add i32 %86, %87
  %89 = icmp ult i32 %88, %26
  br i1 %89, label %90, label %143

90:                                               ; preds = %84
  %91 = extractelement <3 x i32> %9, i64 1
  %92 = icmp ult i32 %91, %14
  br i1 %92, label %93, label %143

93:                                               ; preds = %90
  %94 = extractelement <3 x i32> %9, i64 2
  %95 = icmp ult i32 %94, %18
  br i1 %95, label %96, label %143

96:                                               ; preds = %93
  %97 = zext i32 %91 to i64
  %98 = zext i32 %18 to i64
  %99 = mul nuw i64 %98, %97
  %100 = zext i32 %94 to i64
  %101 = add nuw i64 %99, %100
  br i1 %50, label %105, label %102

102:                                              ; preds = %96
  %103 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %101
  %104 = load i64, i64 addrspace(1)* %103, align 8, !tbaa !54
  br label %105

105:                                              ; preds = %102, %96
  %106 = phi i64 [ %104, %102 ], [ 0, %96 ]
  %107 = icmp sgt i64 %106, -1
  %108 = zext i32 %22 to i64
  %109 = icmp ult i64 %106, %108
  %110 = select i1 %107, i1 %109, i1 false
  br i1 %110, label %131, label %111

111:                                              ; preds = %105
  %112 = icmp eq i32 %11, 0
  br i1 %112, label %113, label %143

113:                                              ; preds = %111
  %114 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %115 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %114, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %116 = zext i32 %26 to i64
  %117 = mul i64 %101, %116
  %118 = zext i32 %88 to i64
  %119 = add i64 %117, %118
  br label %120

120:                                              ; preds = %128, %113
  %121 = phi i32 [ 0, %113 ], [ %129, %128 ]
  %122 = add nuw nsw i32 %121, %88
  %123 = icmp ult i32 %122, %26
  br i1 %123, label %124, label %128

124:                                              ; preds = %120
  %125 = zext i32 %121 to i64
  %126 = add i64 %119, %125
  %127 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %126
  store bfloat 0xR7FC0, bfloat addrspace(1)* %127, align 2, !tbaa !55
  br label %128

128:                                              ; preds = %124, %120
  %129 = add nuw nsw i32 %121, 1
  %130 = icmp eq i32 %129, 4
  br i1 %130, label %143, label %120, !llvm.loop !60

131:                                              ; preds = %105
  %132 = icmp eq i32 %48, 0
  %133 = select i1 %132, i64 %97, i64 %101
  %134 = mul i64 %133, %60
  %135 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %134
  %136 = icmp ugt i32 %26, 7
  %137 = and i32 %30, 511
  %138 = icmp eq i32 %137, 0
  %139 = select i1 %136, i1 %138, i1 false
  %140 = trunc i64 %106 to i32
  br i1 %139, label %141, label %142

141:                                              ; preds = %131
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %88, i32 noundef %140, i64 noundef %101, i32 noundef %11) #12
  br label %143

142:                                              ; preds = %131
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %88, i32 noundef %140, i64 noundef %101, i32 noundef %11) #12
  br label %143

143:                                              ; preds = %142, %141, %128, %111, %93, %90, %84, %81, %79
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_control_probe_q6_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* noundef "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #10
  %17 = icmp ne i32 %11, 32
  %18 = or i1 %17, %16
  %19 = icmp ugt i32 %12, 1
  %20 = or i1 %19, %18
  %21 = icmp ugt i32 %13, 31
  %22 = or i1 %21, %20
  br i1 %22, label %23, label %28

23:                                               ; preds = %14
  %24 = icmp eq i32 %13, 0
  br i1 %24, label %25, label %29

25:                                               ; preds = %23
  %26 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN22r5_raw_odd_control_tap7projectILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap7projectILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceKlPU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS9_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, i64 addrspace(1)* noundef %4, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #1 {
  %13 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 0
  %14 = load i32, i32 addrspace(2)* %13, align 8, !tbaa !38
  %15 = icmp eq i32 %14, 0
  br i1 %15, label %79, label %16

16:                                               ; preds = %12
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 1
  %18 = load i32, i32 addrspace(2)* %17, align 4, !tbaa !44
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %79, label %20

20:                                               ; preds = %16
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 4
  %22 = load i32, i32 addrspace(2)* %21, align 8, !tbaa !45
  %23 = icmp eq i32 %22, 0
  br i1 %23, label %79, label %24

24:                                               ; preds = %20
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !46
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %79, label %28

28:                                               ; preds = %24
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 2
  %30 = load i32, i32 addrspace(2)* %29, align 8, !tbaa !47
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %79, label %32

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 5
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !48
  %35 = icmp eq i32 %34, 6
  br i1 %35, label %36, label %79

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 6
  %38 = load i32, i32 addrspace(2)* %37, align 8, !tbaa !49
  %39 = icmp eq i32 %38, 64
  %40 = and i32 %30, 63
  %41 = icmp eq i32 %40, 0
  %42 = select i1 %39, i1 %41, i1 false
  br i1 %42, label %43, label %79

43:                                               ; preds = %36
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 7
  %45 = load i32, i32 addrspace(2)* %44, align 4, !tbaa !50
  %46 = icmp ult i32 %45, 4
  br i1 %46, label %47, label %79

47:                                               ; preds = %43
  %48 = and i32 %45, 2
  %49 = and i32 %45, 1
  %50 = icmp eq i32 %49, 0
  %51 = icmp eq i32 %45, 2
  br i1 %51, label %79, label %52

52:                                               ; preds = %47
  br i1 %50, label %53, label %57

53:                                               ; preds = %52
  %54 = icmp eq i32 %22, 1
  %55 = icmp eq i32 %18, 1
  %56 = select i1 %54, i1 %55, i1 false
  br i1 %56, label %57, label %79

57:                                               ; preds = %53, %52
  %58 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 8
  %59 = load i64, i64 addrspace(2)* %58, align 8, !tbaa !51
  %60 = zext i32 %30 to i64
  %61 = mul nuw nsw i64 %60, 6
  %62 = lshr i64 %61, 3
  %63 = icmp ult i64 %59, %62
  br i1 %63, label %79, label %64

64:                                               ; preds = %57
  %65 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 10
  %66 = load i64, i64 addrspace(2)* %65, align 8, !tbaa !52
  %67 = lshr i32 %30, 5
  %68 = and i32 %67, 134217726
  %69 = zext i32 %68 to i64
  %70 = icmp uge i64 %66, %69
  %71 = and i64 %66, 1
  %72 = icmp eq i64 %71, 0
  %73 = and i1 %70, %72
  br i1 %73, label %74, label %79

74:                                               ; preds = %64
  %75 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %8, i64 0, i32 11
  %76 = load i64, i64 addrspace(2)* %75, align 8, !tbaa !53
  %77 = and i64 %76, 1
  %78 = icmp eq i64 %77, 0
  br i1 %78, label %84, label %79

79:                                               ; preds = %74, %64, %57, %53, %47, %43, %36, %32, %28, %24, %20, %16, %12
  %80 = icmp eq i32 %11, 0
  br i1 %80, label %81, label %143

81:                                               ; preds = %79
  %82 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %83 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %82, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %143

84:                                               ; preds = %74
  %85 = extractelement <3 x i32> %9, i64 0
  %86 = shl i32 %85, 3
  %87 = shl i32 %10, 2
  %88 = add i32 %86, %87
  %89 = icmp ult i32 %88, %26
  br i1 %89, label %90, label %143

90:                                               ; preds = %84
  %91 = extractelement <3 x i32> %9, i64 1
  %92 = icmp ult i32 %91, %14
  br i1 %92, label %93, label %143

93:                                               ; preds = %90
  %94 = extractelement <3 x i32> %9, i64 2
  %95 = icmp ult i32 %94, %18
  br i1 %95, label %96, label %143

96:                                               ; preds = %93
  %97 = zext i32 %91 to i64
  %98 = zext i32 %18 to i64
  %99 = mul nuw i64 %98, %97
  %100 = zext i32 %94 to i64
  %101 = add nuw i64 %99, %100
  br i1 %50, label %105, label %102

102:                                              ; preds = %96
  %103 = getelementptr inbounds i64, i64 addrspace(1)* %4, i64 %101
  %104 = load i64, i64 addrspace(1)* %103, align 8, !tbaa !54
  br label %105

105:                                              ; preds = %102, %96
  %106 = phi i64 [ %104, %102 ], [ 0, %96 ]
  %107 = icmp sgt i64 %106, -1
  %108 = zext i32 %22 to i64
  %109 = icmp ult i64 %106, %108
  %110 = select i1 %107, i1 %109, i1 false
  br i1 %110, label %131, label %111

111:                                              ; preds = %105
  %112 = icmp eq i32 %11, 0
  br i1 %112, label %113, label %143

113:                                              ; preds = %111
  %114 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %115 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %114, i32 1, i32 0, i32 2, i32 0, i1 false) #11
  %116 = zext i32 %26 to i64
  %117 = mul i64 %101, %116
  %118 = zext i32 %88 to i64
  %119 = add i64 %117, %118
  br label %120

120:                                              ; preds = %128, %113
  %121 = phi i32 [ 0, %113 ], [ %129, %128 ]
  %122 = add nuw nsw i32 %121, %88
  %123 = icmp ult i32 %122, %26
  br i1 %123, label %124, label %128

124:                                              ; preds = %120
  %125 = zext i32 %121 to i64
  %126 = add i64 %119, %125
  %127 = getelementptr inbounds bfloat, bfloat addrspace(1)* %5, i64 %126
  store bfloat 0xR7FC0, bfloat addrspace(1)* %127, align 2, !tbaa !55
  br label %128

128:                                              ; preds = %124, %120
  %129 = add nuw nsw i32 %121, 1
  %130 = icmp eq i32 %129, 4
  br i1 %130, label %143, label %120, !llvm.loop !61

131:                                              ; preds = %105
  %132 = icmp eq i32 %48, 0
  %133 = select i1 %132, i64 %97, i64 %101
  %134 = mul i64 %133, %60
  %135 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %134
  %136 = icmp ugt i32 %26, 7
  %137 = and i32 %30, 255
  %138 = icmp eq i32 %137, 0
  %139 = select i1 %136, i1 %138, i1 false
  %140 = trunc i64 %106 to i32
  br i1 %139, label %141, label %142

141:                                              ; preds = %131
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %88, i32 noundef %140, i64 noundef %101, i32 noundef %11) #12
  br label %143

142:                                              ; preds = %131
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt6ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %135, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %7, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %8, i32 noundef %88, i32 noundef %140, i64 noundef %101, i32 noundef %11) #12
  br label %143

143:                                              ; preds = %142, %141, %128, %111, %93, %90, %84, %81, %79
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i1 @air.any.v3i1(<3 x i1>) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture, i32, i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %24, label %20

20:                                               ; preds = %12
  %21 = shl i32 %11, 4
  %22 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  br label %33

24:                                               ; preds = %73, %12
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !46
  %27 = icmp eq i32 %11, 0
  %28 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %29 = zext i32 %26 to i64
  %30 = mul i64 %29, %10
  %31 = zext i32 %8 to i64
  %32 = add i64 %30, %31
  br label %77

33:                                               ; preds = %73, %20
  %34 = phi i32 [ 0, %20 ], [ %74, %73 ]
  %35 = add i32 %34, %21
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %36
  br label %38

38:                                               ; preds = %38, %33
  %39 = phi i32 [ 0, %33 ], [ %71, %38 ]
  %40 = phi float [ 0.000000e+00, %33 ], [ %63, %38 ]
  %41 = zext i32 %39 to i64
  %42 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %41
  %43 = load bfloat, bfloat addrspace(1)* %42, align 2, !tbaa !55
  %44 = fpext bfloat %43 to float
  %45 = or i32 %39, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !55
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %39, 2
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !55
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = or i32 %39, 3
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %58
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !55
  %61 = fpext bfloat %60 to float
  %62 = fadd float %56, %61
  %63 = fadd float %40, %62
  %64 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %41
  store float %44, float* %64, align 4, !tbaa !62
  %65 = fmul float %49, 6.250000e-02
  %66 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %46
  store float %65, float* %66, align 4, !tbaa !62
  %67 = fmul float %55, 3.906250e-03
  %68 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %52
  store float %67, float* %68, align 4, !tbaa !62
  %69 = fmul float %61, 0x3F30000000000000
  %70 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %58
  store float %69, float* %70, align 4, !tbaa !62
  %71 = add nuw nsw i32 %39, 4
  %72 = icmp ult i32 %39, 12
  br i1 %72, label %38, label %73, !llvm.loop !64

73:                                               ; preds = %38
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %35, i32 noundef %9, float* noundef nonnull %22, float noundef %63, i32 noundef 16, i1 noundef zeroext false, float* noundef nonnull %23) #14
  %74 = add i32 %34, 512
  %75 = icmp ult i32 %74, %18
  br i1 %75, label %33, label %24, !llvm.loop !65

76:                                               ; preds = %102
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #13
  ret void

77:                                               ; preds = %102, %24
  %78 = phi i32 [ 0, %24 ], [ %103, %102 ]
  %79 = add i32 %78, %8
  %80 = icmp ult i32 %79, %26
  br i1 %80, label %81, label %102

81:                                               ; preds = %77
  %82 = zext i32 %78 to i64
  %83 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %82
  %84 = load float, float* %83, align 4, !tbaa !62
  %85 = call fast float @air.simd_sum.f32(float %84) #15
  br i1 %27, label %86, label %102

86:                                               ; preds = %81
  %87 = fptrunc float %85 to bfloat
  %88 = bitcast float %85 to i32
  %89 = and i32 %88, 2139095040
  %90 = icmp eq i32 %89, 2139095040
  br i1 %90, label %96, label %91

91:                                               ; preds = %86
  %92 = fpext bfloat %87 to float
  %93 = bitcast float %92 to i32
  %94 = and i32 %93, 2139095040
  %95 = icmp eq i32 %94, 2139095040
  br i1 %95, label %96, label %98

96:                                               ; preds = %91, %86
  %97 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %28, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %98

98:                                               ; preds = %96, %91
  %99 = add i64 %32, %82
  %100 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %99
  store bfloat %87, bfloat addrspace(1)* %100, align 2, !tbaa !55
  %101 = getelementptr inbounds float, float addrspace(1)* %6, i64 %99
  store float %85, float addrspace(1)* %101, align 4, !tbaa !62
  br label %102

102:                                              ; preds = %98, %81, %77
  %103 = add nuw nsw i32 %78, 1
  %104 = icmp eq i32 %103, 4
  br i1 %104, label %76, label %77, !llvm.loop !66
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt4ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [8 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp ugt i32 %18, 256
  %20 = shl i32 %11, 3
  br i1 %19, label %21, label %69

21:                                               ; preds = %12
  %22 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  br label %24

24:                                               ; preds = %63, %21
  %25 = phi i32 [ 0, %21 ], [ %64, %63 ]
  %26 = add i32 %25, %20
  %27 = zext i32 %26 to i64
  %28 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %27
  br label %29

29:                                               ; preds = %29, %24
  %30 = phi i1 [ true, %24 ], [ false, %29 ]
  %31 = phi i32 [ 0, %24 ], [ 4, %29 ]
  %32 = phi float [ 0.000000e+00, %24 ], [ %55, %29 ]
  %33 = zext i32 %31 to i64
  %34 = getelementptr inbounds bfloat, bfloat addrspace(1)* %28, i64 %33
  %35 = load bfloat, bfloat addrspace(1)* %34, align 2, !tbaa !55
  %36 = fpext bfloat %35 to float
  %37 = or i32 %31, 1
  %38 = zext i32 %37 to i64
  %39 = getelementptr inbounds bfloat, bfloat addrspace(1)* %28, i64 %38
  %40 = load bfloat, bfloat addrspace(1)* %39, align 2, !tbaa !55
  %41 = fpext bfloat %40 to float
  %42 = fadd float %36, %41
  %43 = or i32 %31, 2
  %44 = zext i32 %43 to i64
  %45 = getelementptr inbounds bfloat, bfloat addrspace(1)* %28, i64 %44
  %46 = load bfloat, bfloat addrspace(1)* %45, align 2, !tbaa !55
  %47 = fpext bfloat %46 to float
  %48 = fadd float %42, %47
  %49 = or i32 %31, 3
  %50 = zext i32 %49 to i64
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %28, i64 %50
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !55
  %53 = fpext bfloat %52 to float
  %54 = fadd float %48, %53
  %55 = fadd float %32, %54
  %56 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %33
  store float %36, float* %56, align 4, !tbaa !62
  %57 = fmul float %41, 6.250000e-02
  %58 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %38
  store float %57, float* %58, align 4, !tbaa !62
  %59 = fmul float %47, 3.906250e-03
  %60 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %44
  store float %59, float* %60, align 4, !tbaa !62
  %61 = fmul float %53, 0x3F30000000000000
  %62 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %50
  store float %61, float* %62, align 4, !tbaa !62
  br i1 %30, label %29, label %63, !llvm.loop !67

63:                                               ; preds = %29
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %26, i32 noundef %9, float* noundef nonnull %22, float noundef %55, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %23) #14
  %64 = add i32 %25, 256
  %65 = icmp ugt i32 %18, %64
  %66 = sub i32 %18, %64
  %67 = icmp ugt i32 %66, 256
  %68 = and i1 %65, %67
  br i1 %68, label %24, label %69, !llvm.loop !68

69:                                               ; preds = %63, %12
  %70 = phi i32 [ 0, %12 ], [ %64, %63 ]
  %71 = phi i32 [ %18, %12 ], [ %66, %63 ]
  %72 = icmp ugt i32 %71, %20
  br i1 %72, label %73, label %76

73:                                               ; preds = %69
  %74 = sub i32 %71, %20
  %75 = call i32 @air.min.u.i32(i32 %74, i32 8) #10
  br label %76

76:                                               ; preds = %73, %69
  %77 = phi i32 [ %75, %73 ], [ 0, %69 ]
  %78 = icmp eq i32 %77, 0
  br i1 %78, label %131, label %79

79:                                               ; preds = %76
  %80 = add i32 %70, %20
  %81 = zext i32 %80 to i64
  %82 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %81
  %83 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %84 = icmp sgt i32 %77, 0
  br i1 %84, label %88, label %85

85:                                               ; preds = %88, %79
  %86 = phi float [ 0.000000e+00, %79 ], [ %113, %88 ]
  %87 = icmp slt i32 %77, 8
  br i1 %87, label %123, label %129

88:                                               ; preds = %88, %79
  %89 = phi i32 [ %121, %88 ], [ 0, %79 ]
  %90 = phi float [ %113, %88 ], [ 0.000000e+00, %79 ]
  %91 = zext i32 %89 to i64
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %82, i64 %91
  %93 = load bfloat, bfloat addrspace(1)* %92, align 2, !tbaa !55
  %94 = fpext bfloat %93 to float
  %95 = or i32 %89, 1
  %96 = zext i32 %95 to i64
  %97 = getelementptr inbounds bfloat, bfloat addrspace(1)* %82, i64 %96
  %98 = load bfloat, bfloat addrspace(1)* %97, align 2, !tbaa !55
  %99 = fpext bfloat %98 to float
  %100 = fadd float %94, %99
  %101 = or i32 %89, 2
  %102 = zext i32 %101 to i64
  %103 = getelementptr inbounds bfloat, bfloat addrspace(1)* %82, i64 %102
  %104 = load bfloat, bfloat addrspace(1)* %103, align 2, !tbaa !55
  %105 = fpext bfloat %104 to float
  %106 = fadd float %100, %105
  %107 = or i32 %89, 3
  %108 = zext i32 %107 to i64
  %109 = getelementptr inbounds bfloat, bfloat addrspace(1)* %82, i64 %108
  %110 = load bfloat, bfloat addrspace(1)* %109, align 2, !tbaa !55
  %111 = fpext bfloat %110 to float
  %112 = fadd float %106, %111
  %113 = fadd float %90, %112
  %114 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %91
  store float %94, float* %114, align 4, !tbaa !62
  %115 = fmul float %99, 6.250000e-02
  %116 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %96
  store float %115, float* %116, align 4, !tbaa !62
  %117 = fmul float %105, 3.906250e-03
  %118 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %102
  store float %117, float* %118, align 4, !tbaa !62
  %119 = fmul float %111, 0x3F30000000000000
  %120 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %108
  store float %119, float* %120, align 4, !tbaa !62
  %121 = add nuw nsw i32 %89, 4
  %122 = icmp slt i32 %121, %77
  br i1 %122, label %88, label %85, !llvm.loop !69

123:                                              ; preds = %123, %85
  %124 = phi i32 [ %127, %123 ], [ %77, %85 ]
  %125 = sext i32 %124 to i64
  %126 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %125
  store float 0.000000e+00, float* %126, align 4, !tbaa !62
  %127 = add i32 %124, 1
  %128 = icmp eq i32 %127, 8
  br i1 %128, label %129, label %123, !llvm.loop !70

129:                                              ; preds = %123, %85
  %130 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %80, i32 noundef %9, float* noundef nonnull %83, float noundef %86, i32 noundef %77, i1 noundef zeroext true, float* noundef nonnull %130) #14
  br label %131

131:                                              ; preds = %129, %76
  %132 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %133 = load i32, i32 addrspace(2)* %132, align 4, !tbaa !46
  %134 = icmp eq i32 %11, 0
  %135 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %136 = zext i32 %133 to i64
  %137 = mul i64 %136, %10
  %138 = zext i32 %8 to i64
  %139 = add i64 %137, %138
  br label %141

140:                                              ; preds = %166
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #13
  ret void

141:                                              ; preds = %166, %131
  %142 = phi i32 [ 0, %131 ], [ %167, %166 ]
  %143 = add i32 %142, %8
  %144 = icmp ult i32 %143, %133
  br i1 %144, label %145, label %166

145:                                              ; preds = %141
  %146 = zext i32 %142 to i64
  %147 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !62
  %149 = call fast float @air.simd_sum.f32(float %148) #15
  br i1 %134, label %150, label %166

150:                                              ; preds = %145
  %151 = fptrunc float %149 to bfloat
  %152 = bitcast float %149 to i32
  %153 = and i32 %152, 2139095040
  %154 = icmp eq i32 %153, 2139095040
  br i1 %154, label %160, label %155

155:                                              ; preds = %150
  %156 = fpext bfloat %151 to float
  %157 = bitcast float %156 to i32
  %158 = and i32 %157, 2139095040
  %159 = icmp eq i32 %158, 2139095040
  br i1 %159, label %160, label %162

160:                                              ; preds = %155, %150
  %161 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %135, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %162

162:                                              ; preds = %160, %155
  %163 = add i64 %139, %146
  %164 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %163
  store bfloat %151, bfloat addrspace(1)* %164, align 2, !tbaa !55
  %165 = getelementptr inbounds float, float addrspace(1)* %6, i64 %163
  store float %149, float addrspace(1)* %165, align 4, !tbaa !62
  br label %166

166:                                              ; preds = %162, %145, %141
  %167 = add nuw nsw i32 %142, 1
  %168 = icmp eq i32 %167, 4
  br i1 %168, label %140, label %141, !llvm.loop !71
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #6

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #7 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !53
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !72
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !46
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
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !55
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !55
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
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !73
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !73
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !62
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !62
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #10
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #13
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !62
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #10
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #13
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !62
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #10
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #13
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !74

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #13
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !62
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !62
  br label %161

109:                                              ; preds = %109, %39
  %110 = phi float [ %151, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i32 [ %152, %109 ], [ 0, %39 ]
  %112 = shl nuw nsw i32 %111, 1
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %113
  %115 = load i8, i8 addrspace(1)* %114, align 1, !tbaa !73
  %116 = zext i8 %115 to i32
  %117 = or i32 %112, 1
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %118
  %120 = load i8, i8 addrspace(1)* %119, align 1, !tbaa !73
  %121 = zext i8 %120 to i32
  %122 = shl nuw nsw i32 %121, 8
  %123 = shl nuw nsw i32 %111, 2
  %124 = zext i32 %123 to i64
  %125 = getelementptr inbounds float, float* %7, i64 %124
  %126 = load float, float* %125, align 4, !tbaa !62
  %127 = and i32 %116, 15
  %128 = tail call float @air.convert.f.f32.s.i32(i32 %127) #10
  %129 = or i32 %123, 1
  %130 = zext i32 %129 to i64
  %131 = getelementptr inbounds float, float* %7, i64 %130
  %132 = load float, float* %131, align 4, !tbaa !62
  %133 = and i32 %116, 240
  %134 = tail call float @air.convert.f.f32.s.i32(i32 %133) #10
  %135 = fmul float %132, %134
  %136 = tail call float @llvm.fmuladd.f32(float %126, float %128, float %135) #13
  %137 = or i32 %123, 2
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds float, float* %7, i64 %138
  %140 = load float, float* %139, align 4, !tbaa !62
  %141 = and i32 %122, 3840
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #10
  %143 = tail call float @llvm.fmuladd.f32(float %140, float %142, float %136) #13
  %144 = or i32 %123, 3
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds float, float* %7, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !62
  %148 = and i32 %122, 61440
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #10
  %150 = tail call float @llvm.fmuladd.f32(float %147, float %149, float %143) #13
  %151 = fadd float %110, %150
  %152 = add nuw nsw i32 %111, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %154, label %109, !llvm.loop !75

154:                                              ; preds = %109
  %155 = fmul float %54, %8
  %156 = tail call float @llvm.fmuladd.f32(float %51, float %151, float %155) #13
  %157 = zext i32 %36 to i64
  %158 = getelementptr inbounds float, float* %11, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !62
  %160 = fadd float %159, %156
  store float %160, float* %158, align 4, !tbaa !62
  br label %161

161:                                              ; preds = %154, %101, %35
  %162 = add nuw nsw i32 %36, 1
  %163 = icmp eq i32 %162, 4
  br i1 %163, label %34, label %35, !llvm.loop !76
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #8

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #9

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #7 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !53
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !72
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !46
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
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !55
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !55
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
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !73
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !73
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !62
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !62
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #10
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #13
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !62
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #10
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #13
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !62
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #10
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #13
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !77

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #13
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !62
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !62
  br label %160

109:                                              ; preds = %109, %39
  %110 = phi float [ %152, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i1 [ false, %109 ], [ true, %39 ]
  %112 = phi i32 [ 1, %109 ], [ 0, %39 ]
  %113 = shl nuw nsw i32 %112, 1
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %114
  %116 = load i8, i8 addrspace(1)* %115, align 1, !tbaa !73
  %117 = zext i8 %116 to i32
  %118 = or i32 %113, 1
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %119
  %121 = load i8, i8 addrspace(1)* %120, align 1, !tbaa !73
  %122 = zext i8 %121 to i32
  %123 = shl nuw nsw i32 %122, 8
  %124 = shl nuw nsw i32 %112, 2
  %125 = zext i32 %124 to i64
  %126 = getelementptr inbounds float, float* %7, i64 %125
  %127 = load float, float* %126, align 4, !tbaa !62
  %128 = and i32 %117, 15
  %129 = tail call float @air.convert.f.f32.s.i32(i32 %128) #10
  %130 = or i32 %124, 1
  %131 = zext i32 %130 to i64
  %132 = getelementptr inbounds float, float* %7, i64 %131
  %133 = load float, float* %132, align 4, !tbaa !62
  %134 = and i32 %117, 240
  %135 = tail call float @air.convert.f.f32.s.i32(i32 %134) #10
  %136 = fmul float %133, %135
  %137 = tail call float @llvm.fmuladd.f32(float %127, float %129, float %136) #13
  %138 = or i32 %124, 2
  %139 = zext i32 %138 to i64
  %140 = getelementptr inbounds float, float* %7, i64 %139
  %141 = load float, float* %140, align 4, !tbaa !62
  %142 = and i32 %123, 3840
  %143 = tail call float @air.convert.f.f32.s.i32(i32 %142) #10
  %144 = tail call float @llvm.fmuladd.f32(float %141, float %143, float %137) #13
  %145 = or i32 %124, 3
  %146 = zext i32 %145 to i64
  %147 = getelementptr inbounds float, float* %7, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !62
  %149 = and i32 %123, 61440
  %150 = tail call float @air.convert.f.f32.s.i32(i32 %149) #10
  %151 = tail call float @llvm.fmuladd.f32(float %148, float %150, float %144) #13
  %152 = fadd float %110, %151
  br i1 %111, label %109, label %153, !llvm.loop !78

153:                                              ; preds = %109
  %154 = fmul float %54, %8
  %155 = tail call float @llvm.fmuladd.f32(float %51, float %152, float %154) #13
  %156 = zext i32 %36 to i64
  %157 = getelementptr inbounds float, float* %11, i64 %156
  %158 = load float, float* %157, align 4, !tbaa !62
  %159 = fadd float %158, %155
  store float %159, float* %157, align 4, !tbaa !62
  br label %160

160:                                              ; preds = %153, %101, %35
  %161 = add nuw nsw i32 %36, 1
  %162 = icmp eq i32 %161, 4
  br i1 %162, label %34, label %35, !llvm.loop !79
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i32 @air.min.u.i32(i32, i32) local_unnamed_addr #2

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %23

20:                                               ; preds = %12
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !46
  br label %40

23:                                               ; preds = %12
  %24 = shl i32 %11, 4
  %25 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %26 = zext i32 %9 to i64
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %28 = load i64, i64 addrspace(2)* %27, align 8, !tbaa !53
  %29 = mul i64 %28, %26
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %31 = load i64, i64 addrspace(2)* %30, align 8, !tbaa !72
  %32 = mul i64 %31, %26
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !46
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %32
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %37 = load i64, i64 addrspace(2)* %36, align 8
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %39 = load i64, i64 addrspace(2)* %38, align 8
  br label %48

40:                                               ; preds = %153, %20
  %41 = phi i32 [ %22, %20 ], [ %34, %153 ]
  %42 = icmp eq i32 %11, 0
  %43 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %44 = zext i32 %41 to i64
  %45 = mul i64 %44, %10
  %46 = zext i32 %8 to i64
  %47 = add i64 %45, %46
  br label %157

48:                                               ; preds = %153, %23
  %49 = phi i32 [ 0, %23 ], [ %154, %153 ]
  %50 = add i32 %49, %24
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %51
  br label %53

53:                                               ; preds = %53, %48
  %54 = phi i1 [ true, %48 ], [ false, %53 ]
  %55 = phi i32 [ 0, %48 ], [ 8, %53 ]
  %56 = phi float [ 0.000000e+00, %48 ], [ %103, %53 ]
  %57 = zext i32 %55 to i64
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %57
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !55
  %60 = fpext bfloat %59 to float
  %61 = or i32 %55, 1
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %62
  %64 = load bfloat, bfloat addrspace(1)* %63, align 2, !tbaa !55
  %65 = fpext bfloat %64 to float
  %66 = fadd float %60, %65
  %67 = or i32 %55, 2
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %68
  %70 = load bfloat, bfloat addrspace(1)* %69, align 2, !tbaa !55
  %71 = fpext bfloat %70 to float
  %72 = fadd float %66, %71
  %73 = or i32 %55, 3
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %74
  %76 = load bfloat, bfloat addrspace(1)* %75, align 2, !tbaa !55
  %77 = fpext bfloat %76 to float
  %78 = fadd float %72, %77
  %79 = or i32 %55, 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %80
  %82 = load bfloat, bfloat addrspace(1)* %81, align 2, !tbaa !55
  %83 = fpext bfloat %82 to float
  %84 = fadd float %78, %83
  %85 = or i32 %55, 5
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %86
  %88 = load bfloat, bfloat addrspace(1)* %87, align 2, !tbaa !55
  %89 = fpext bfloat %88 to float
  %90 = fadd float %84, %89
  %91 = or i32 %55, 6
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %92
  %94 = load bfloat, bfloat addrspace(1)* %93, align 2, !tbaa !55
  %95 = fpext bfloat %94 to float
  %96 = fadd float %90, %95
  %97 = or i32 %55, 7
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %98
  %100 = load bfloat, bfloat addrspace(1)* %99, align 2, !tbaa !55
  %101 = fpext bfloat %100 to float
  %102 = fadd float %96, %101
  %103 = fadd float %56, %102
  %104 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %57
  store float %60, float* %104, align 4, !tbaa !62
  %105 = fmul float %65, 3.125000e-02
  %106 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %62
  store float %105, float* %106, align 4, !tbaa !62
  %107 = fmul float %71, 2.500000e-01
  %108 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %68
  store float %107, float* %108, align 4, !tbaa !62
  %109 = fmul float %77, 7.812500e-03
  %110 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %74
  store float %109, float* %110, align 4, !tbaa !62
  %111 = fmul float %83, 6.250000e-02
  %112 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %80
  store float %111, float* %112, align 4, !tbaa !62
  %113 = fmul float %89, 5.000000e-01
  %114 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %86
  store float %113, float* %114, align 4, !tbaa !62
  %115 = fmul float %95, 1.562500e-02
  %116 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %92
  store float %115, float* %116, align 4, !tbaa !62
  %117 = fmul float %101, 1.250000e-01
  %118 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %98
  store float %117, float* %118, align 4, !tbaa !62
  br i1 %54, label %53, label %119, !llvm.loop !80

119:                                              ; preds = %53
  %120 = lshr i32 %50, 6
  %121 = mul nuw nsw i64 %51, 5
  %122 = lshr exact i64 %121, 3
  %123 = zext i32 %120 to i64
  %124 = getelementptr inbounds i8, i8 addrspace(1)* %35, i64 %122
  br label %125

125:                                              ; preds = %150, %119
  %126 = phi i32 [ 0, %119 ], [ %151, %150 ]
  %127 = add i32 %126, %8
  %128 = icmp ult i32 %127, %34
  br i1 %128, label %129, label %150

129:                                              ; preds = %125
  %130 = zext i32 %127 to i64
  %131 = mul i64 %37, %130
  %132 = getelementptr inbounds i8, i8 addrspace(1)* %124, i64 %131
  %133 = mul i64 %39, %130
  %134 = add i64 %133, %29
  %135 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %134
  %136 = bitcast i8 addrspace(1)* %135 to bfloat addrspace(1)*
  %137 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %134
  %138 = bitcast i8 addrspace(1)* %137 to bfloat addrspace(1)*
  %139 = getelementptr inbounds bfloat, bfloat addrspace(1)* %136, i64 %123
  %140 = load bfloat, bfloat addrspace(1)* %139, align 2, !tbaa !55
  %141 = fpext bfloat %140 to float
  %142 = getelementptr inbounds bfloat, bfloat addrspace(1)* %138, i64 %123
  %143 = load bfloat, bfloat addrspace(1)* %142, align 2, !tbaa !55
  %144 = fpext bfloat %143 to float
  %145 = call fast float @_ZN22r5_raw_odd_control_tap23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %132, float* noundef nonnull %25, float noundef %141, float noundef %144, float noundef %103) #16
  %146 = zext i32 %126 to i64
  %147 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !62
  %149 = fadd float %145, %148
  store float %149, float* %147, align 4, !tbaa !62
  br label %150

150:                                              ; preds = %129, %125
  %151 = add nuw nsw i32 %126, 1
  %152 = icmp eq i32 %151, 4
  br i1 %152, label %153, label %125, !llvm.loop !81

153:                                              ; preds = %150
  %154 = add i32 %49, 512
  %155 = icmp ult i32 %154, %18
  br i1 %155, label %48, label %40, !llvm.loop !82

156:                                              ; preds = %182
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #13
  ret void

157:                                              ; preds = %182, %40
  %158 = phi i32 [ 0, %40 ], [ %183, %182 ]
  %159 = add i32 %158, %8
  %160 = icmp ult i32 %159, %41
  br i1 %160, label %161, label %182

161:                                              ; preds = %157
  %162 = zext i32 %158 to i64
  %163 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %162
  %164 = load float, float* %163, align 4, !tbaa !62
  %165 = call fast float @air.simd_sum.f32(float %164) #15
  br i1 %42, label %166, label %182

166:                                              ; preds = %161
  %167 = fptrunc float %165 to bfloat
  %168 = bitcast float %165 to i32
  %169 = and i32 %168, 2139095040
  %170 = icmp eq i32 %169, 2139095040
  br i1 %170, label %176, label %171

171:                                              ; preds = %166
  %172 = fpext bfloat %167 to float
  %173 = bitcast float %172 to i32
  %174 = and i32 %173, 2139095040
  %175 = icmp eq i32 %174, 2139095040
  br i1 %175, label %176, label %178

176:                                              ; preds = %171, %166
  %177 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %43, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %178

178:                                              ; preds = %176, %171
  %179 = add i64 %47, %162
  %180 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %179
  store bfloat %167, bfloat addrspace(1)* %180, align 2, !tbaa !55
  %181 = getelementptr inbounds float, float addrspace(1)* %6, i64 %179
  store float %165, float addrspace(1)* %181, align 4, !tbaa !62
  br label %182

182:                                              ; preds = %178, %161, %157
  %183 = add nuw nsw i32 %158, 1
  %184 = icmp eq i32 %183, 4
  br i1 %184, label %156, label %157, !llvm.loop !83
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [8 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp ugt i32 %18, 256
  %20 = shl i32 %11, 3
  br i1 %19, label %21, label %182

21:                                               ; preds = %12
  %22 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 2
  %24 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 4
  %25 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 6
  %26 = zext i32 %9 to i64
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %28 = load i64, i64 addrspace(2)* %27, align 8, !tbaa !53
  %29 = mul i64 %28, %26
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %31 = load i64, i64 addrspace(2)* %30, align 8, !tbaa !72
  %32 = mul i64 %31, %26
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !46
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %32
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %37 = load i64, i64 addrspace(2)* %36, align 8
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %39 = load i64, i64 addrspace(2)* %38, align 8
  br label %40

40:                                               ; preds = %171, %21
  %41 = phi i32 [ 0, %21 ], [ %172, %171 ]
  %42 = add i32 %41, %20
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !55
  %46 = fpext bfloat %45 to float
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 1
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !55
  %49 = fpext bfloat %48 to float
  %50 = fadd float %46, %49
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 2
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !55
  %53 = fpext bfloat %52 to float
  %54 = fadd float %50, %53
  %55 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 3
  %56 = load bfloat, bfloat addrspace(1)* %55, align 2, !tbaa !55
  %57 = fpext bfloat %56 to float
  %58 = fadd float %54, %57
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 4
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !55
  %61 = fpext bfloat %60 to float
  %62 = fadd float %58, %61
  %63 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 5
  %64 = load bfloat, bfloat addrspace(1)* %63, align 2, !tbaa !55
  %65 = fpext bfloat %64 to float
  %66 = fadd float %62, %65
  %67 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 6
  %68 = load bfloat, bfloat addrspace(1)* %67, align 2, !tbaa !55
  %69 = fpext bfloat %68 to float
  %70 = fadd float %66, %69
  %71 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 7
  %72 = load bfloat, bfloat addrspace(1)* %71, align 2, !tbaa !55
  %73 = fpext bfloat %72 to float
  %74 = fadd float %70, %73
  %75 = fadd float %74, 0.000000e+00
  %76 = fmul float %49, 3.125000e-02
  %77 = fmul float %53, 2.500000e-01
  %78 = fmul float %57, 7.812500e-03
  %79 = fmul float %61, 6.250000e-02
  %80 = fmul float %65, 5.000000e-01
  %81 = fmul float %69, 1.562500e-02
  %82 = fmul float %73, 1.250000e-01
  %83 = lshr i32 %42, 6
  %84 = mul nuw nsw i64 %43, 5
  %85 = lshr exact i64 %84, 3
  %86 = zext i32 %83 to i64
  %87 = getelementptr inbounds i8, i8 addrspace(1)* %35, i64 %85
  %88 = fmul float %76, 2.560000e+02
  %89 = fmul float %78, 2.560000e+02
  %90 = fmul float %79, 2.560000e+02
  %91 = fmul float %81, 2.560000e+02
  br label %92

92:                                               ; preds = %168, %40
  %93 = phi i32 [ 0, %40 ], [ %169, %168 ]
  %94 = add i32 %93, %8
  %95 = icmp ult i32 %94, %34
  br i1 %95, label %96, label %168

96:                                               ; preds = %92
  %97 = zext i32 %94 to i64
  %98 = mul i64 %37, %97
  %99 = getelementptr inbounds i8, i8 addrspace(1)* %87, i64 %98
  %100 = mul i64 %39, %97
  %101 = add i64 %100, %29
  %102 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %101
  %103 = bitcast i8 addrspace(1)* %102 to bfloat addrspace(1)*
  %104 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %101
  %105 = bitcast i8 addrspace(1)* %104 to bfloat addrspace(1)*
  %106 = getelementptr inbounds bfloat, bfloat addrspace(1)* %103, i64 %86
  %107 = load bfloat, bfloat addrspace(1)* %106, align 2, !tbaa !55
  %108 = fpext bfloat %107 to float
  %109 = getelementptr inbounds bfloat, bfloat addrspace(1)* %105, i64 %86
  %110 = load bfloat, bfloat addrspace(1)* %109, align 2, !tbaa !55
  %111 = fpext bfloat %110 to float
  %112 = load i8, i8 addrspace(1)* %99, align 1, !tbaa !73
  %113 = zext i8 %112 to i32
  %114 = and i32 %113, 31
  %115 = tail call float @air.convert.f.f32.s.i32(i32 %114) #10
  %116 = and i32 %113, 224
  %117 = tail call float @air.convert.f.f32.s.i32(i32 %116) #10
  %118 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 1
  %119 = load i8, i8 addrspace(1)* %118, align 1, !tbaa !73
  %120 = zext i8 %119 to i32
  %121 = and i32 %120, 3
  %122 = tail call float @air.convert.f.f32.s.i32(i32 %121) #10
  %123 = and i32 %120, 124
  %124 = tail call float @air.convert.f.f32.s.i32(i32 %123) #10
  %125 = and i32 %120, 128
  %126 = tail call float @air.convert.f.f32.s.i32(i32 %125) #10
  %127 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 2
  %128 = load i8, i8 addrspace(1)* %127, align 1, !tbaa !73
  %129 = zext i8 %128 to i32
  %130 = and i32 %129, 15
  %131 = tail call float @air.convert.f.f32.s.i32(i32 %130) #10
  %132 = and i32 %129, 240
  %133 = tail call float @air.convert.f.f32.s.i32(i32 %132) #10
  %134 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 3
  %135 = load i8, i8 addrspace(1)* %134, align 1, !tbaa !73
  %136 = zext i8 %135 to i32
  %137 = and i32 %136, 1
  %138 = tail call float @air.convert.f.f32.s.i32(i32 %137) #10
  %139 = and i32 %136, 62
  %140 = tail call float @air.convert.f.f32.s.i32(i32 %139) #10
  %141 = and i32 %136, 192
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #10
  %143 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 4
  %144 = load i8, i8 addrspace(1)* %143, align 1, !tbaa !73
  %145 = zext i8 %144 to i32
  %146 = and i32 %145, 7
  %147 = tail call float @air.convert.f.f32.s.i32(i32 %146) #10
  %148 = and i32 %145, 248
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #10
  %150 = tail call float @llvm.fmuladd.f32(float %115, float %46, float 0.000000e+00) #13
  %151 = tail call float @llvm.fmuladd.f32(float %117, float %76, float %150) #13
  %152 = tail call float @llvm.fmuladd.f32(float %122, float %88, float %151) #13
  %153 = tail call float @llvm.fmuladd.f32(float %124, float %77, float %152) #13
  %154 = tail call float @llvm.fmuladd.f32(float %126, float %78, float %153) #13
  %155 = tail call float @llvm.fmuladd.f32(float %131, float %89, float %154) #13
  %156 = tail call float @llvm.fmuladd.f32(float %133, float %79, float %155) #13
  %157 = tail call float @llvm.fmuladd.f32(float %138, float %90, float %156) #13
  %158 = tail call float @llvm.fmuladd.f32(float %140, float %80, float %157) #13
  %159 = tail call float @llvm.fmuladd.f32(float %142, float %81, float %158) #13
  %160 = tail call float @llvm.fmuladd.f32(float %147, float %91, float %159) #13
  %161 = tail call float @llvm.fmuladd.f32(float %149, float %82, float %160) #13
  %162 = fmul float %75, %111
  %163 = tail call float @llvm.fmuladd.f32(float %108, float %161, float %162) #13
  %164 = zext i32 %93 to i64
  %165 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %164
  %166 = load float, float* %165, align 4, !tbaa !62
  %167 = fadd float %166, %163
  store float %167, float* %165, align 4, !tbaa !62
  br label %168

168:                                              ; preds = %96, %92
  %169 = add nuw nsw i32 %93, 1
  %170 = icmp eq i32 %169, 4
  br i1 %170, label %171, label %92, !llvm.loop !84

171:                                              ; preds = %168
  %172 = add i32 %41, 256
  %173 = icmp ugt i32 %18, %172
  %174 = sub i32 %18, %172
  %175 = icmp ugt i32 %174, 256
  %176 = and i1 %173, %175
  br i1 %176, label %40, label %177, !llvm.loop !85

177:                                              ; preds = %171
  %178 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 1
  %179 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 3
  %180 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 5
  %181 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 7
  store float %46, float* %22, align 4, !tbaa !62
  store float %76, float* %178, align 4, !tbaa !62
  store float %77, float* %23, align 4, !tbaa !62
  store float %78, float* %179, align 4, !tbaa !62
  store float %79, float* %24, align 4, !tbaa !62
  store float %80, float* %180, align 4, !tbaa !62
  store float %81, float* %25, align 4, !tbaa !62
  store float %82, float* %181, align 4, !tbaa !62
  br label %182

182:                                              ; preds = %177, %12
  %183 = phi i32 [ %172, %177 ], [ 0, %12 ]
  %184 = phi i32 [ %174, %177 ], [ %18, %12 ]
  %185 = icmp ugt i32 %184, %20
  br i1 %185, label %186, label %189

186:                                              ; preds = %182
  %187 = sub i32 %184, %20
  %188 = tail call i32 @air.min.u.i32(i32 %187, i32 8) #10
  br label %189

189:                                              ; preds = %186, %182
  %190 = phi i32 [ %188, %186 ], [ 0, %182 ]
  %191 = icmp eq i32 %190, 0
  br i1 %191, label %192, label %195

192:                                              ; preds = %189
  %193 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %194 = load i32, i32 addrspace(2)* %193, align 4, !tbaa !46
  br label %248

195:                                              ; preds = %189
  %196 = add i32 %183, %20
  %197 = zext i32 %196 to i64
  %198 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %197
  %199 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %200 = call fast float @_ZN22r5_raw_odd_control_tap35mlx_qmv_f32xsum_v1_load_vector_safeIDF16bfLi8ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_i(bfloat addrspace(1)* noundef %198, float* noundef nonnull %199, i32 noundef %190) #14
  %201 = zext i32 %9 to i64
  %202 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %203 = load i64, i64 addrspace(2)* %202, align 8, !tbaa !53
  %204 = mul i64 %203, %201
  %205 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %206 = load i64, i64 addrspace(2)* %205, align 8, !tbaa !72
  %207 = mul i64 %206, %201
  %208 = lshr i32 %196, 6
  %209 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %210 = load i32, i32 addrspace(2)* %209, align 4, !tbaa !46
  %211 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %207
  %212 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %213 = load i64, i64 addrspace(2)* %212, align 8
  %214 = mul nuw nsw i64 %197, 5
  %215 = lshr exact i64 %214, 3
  %216 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %217 = load i64, i64 addrspace(2)* %216, align 8
  %218 = zext i32 %208 to i64
  %219 = getelementptr inbounds i8, i8 addrspace(1)* %211, i64 %215
  br label %220

220:                                              ; preds = %245, %195
  %221 = phi i32 [ 0, %195 ], [ %246, %245 ]
  %222 = add i32 %221, %8
  %223 = icmp ult i32 %222, %210
  br i1 %223, label %224, label %245

224:                                              ; preds = %220
  %225 = zext i32 %222 to i64
  %226 = mul i64 %213, %225
  %227 = getelementptr inbounds i8, i8 addrspace(1)* %219, i64 %226
  %228 = mul i64 %217, %225
  %229 = add i64 %228, %204
  %230 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %229
  %231 = bitcast i8 addrspace(1)* %230 to bfloat addrspace(1)*
  %232 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %229
  %233 = bitcast i8 addrspace(1)* %232 to bfloat addrspace(1)*
  %234 = getelementptr inbounds bfloat, bfloat addrspace(1)* %231, i64 %218
  %235 = load bfloat, bfloat addrspace(1)* %234, align 2, !tbaa !55
  %236 = fpext bfloat %235 to float
  %237 = getelementptr inbounds bfloat, bfloat addrspace(1)* %233, i64 %218
  %238 = load bfloat, bfloat addrspace(1)* %237, align 2, !tbaa !55
  %239 = fpext bfloat %238 to float
  %240 = call fast float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %227, float* noundef nonnull %199, float noundef %236, float noundef %239, float noundef %200, i32 noundef %190) #16
  %241 = zext i32 %221 to i64
  %242 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %241
  %243 = load float, float* %242, align 4, !tbaa !62
  %244 = fadd float %240, %243
  store float %244, float* %242, align 4, !tbaa !62
  br label %245

245:                                              ; preds = %224, %220
  %246 = add nuw nsw i32 %221, 1
  %247 = icmp eq i32 %246, 4
  br i1 %247, label %248, label %220, !llvm.loop !84

248:                                              ; preds = %245, %192
  %249 = phi i32 [ %194, %192 ], [ %210, %245 ]
  %250 = icmp eq i32 %11, 0
  %251 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %252 = zext i32 %249 to i64
  %253 = mul i64 %252, %10
  %254 = zext i32 %8 to i64
  %255 = add i64 %253, %254
  br label %257

256:                                              ; preds = %282
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #13
  ret void

257:                                              ; preds = %282, %248
  %258 = phi i32 [ 0, %248 ], [ %283, %282 ]
  %259 = add i32 %258, %8
  %260 = icmp ult i32 %259, %249
  br i1 %260, label %261, label %282

261:                                              ; preds = %257
  %262 = zext i32 %258 to i64
  %263 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %262
  %264 = load float, float* %263, align 4, !tbaa !62
  %265 = call fast float @air.simd_sum.f32(float %264) #15
  br i1 %250, label %266, label %282

266:                                              ; preds = %261
  %267 = fptrunc float %265 to bfloat
  %268 = bitcast float %265 to i32
  %269 = and i32 %268, 2139095040
  %270 = icmp eq i32 %269, 2139095040
  br i1 %270, label %276, label %271

271:                                              ; preds = %266
  %272 = fpext bfloat %267 to float
  %273 = bitcast float %272 to i32
  %274 = and i32 %273, 2139095040
  %275 = icmp eq i32 %274, 2139095040
  br i1 %275, label %276, label %278

276:                                              ; preds = %271, %266
  %277 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %251, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %278

278:                                              ; preds = %276, %271
  %279 = add i64 %255, %262
  %280 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %279
  store bfloat %267, bfloat addrspace(1)* %280, align 2, !tbaa !55
  %281 = getelementptr inbounds float, float addrspace(1)* %6, i64 %279
  store float %265, float addrspace(1)* %281, align 4, !tbaa !62
  br label %282

282:                                              ; preds = %278, %261, %257
  %283 = add nuw nsw i32 %258, 1
  %284 = icmp eq i32 %283, 4
  br i1 %284, label %256, label %257, !llvm.loop !86
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4) local_unnamed_addr #7 {
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
  %21 = load i8, i8 addrspace(1)* %20, align 1, !tbaa !73
  %22 = zext i8 %21 to i32
  %23 = and i32 %22, 31
  %24 = tail call float @air.convert.f.f32.s.i32(i32 %23) #10
  %25 = load float, float* %17, align 4, !tbaa !62
  %26 = tail call float @llvm.fmuladd.f32(float %24, float %25, float %12)
  %27 = and i32 %22, 224
  %28 = tail call float @air.convert.f.f32.s.i32(i32 %27) #10
  %29 = getelementptr inbounds float, float* %17, i64 1
  %30 = load float, float* %29, align 4, !tbaa !62
  %31 = tail call float @llvm.fmuladd.f32(float %28, float %30, float %26)
  %32 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 1
  %33 = load i8, i8 addrspace(1)* %32, align 1, !tbaa !73
  %34 = zext i8 %33 to i32
  %35 = and i32 %34, 3
  %36 = tail call float @air.convert.f.f32.s.i32(i32 %35) #10
  %37 = fmul float %30, 2.560000e+02
  %38 = tail call float @llvm.fmuladd.f32(float %36, float %37, float %31)
  %39 = and i32 %34, 124
  %40 = tail call float @air.convert.f.f32.s.i32(i32 %39) #10
  %41 = getelementptr inbounds float, float* %17, i64 2
  %42 = load float, float* %41, align 4, !tbaa !62
  %43 = tail call float @llvm.fmuladd.f32(float %40, float %42, float %38)
  %44 = and i32 %34, 128
  %45 = tail call float @air.convert.f.f32.s.i32(i32 %44) #10
  %46 = getelementptr inbounds float, float* %17, i64 3
  %47 = load float, float* %46, align 4, !tbaa !62
  %48 = tail call float @llvm.fmuladd.f32(float %45, float %47, float %43)
  %49 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 2
  %50 = load i8, i8 addrspace(1)* %49, align 1, !tbaa !73
  %51 = zext i8 %50 to i32
  %52 = and i32 %51, 15
  %53 = tail call float @air.convert.f.f32.s.i32(i32 %52) #10
  %54 = fmul float %47, 2.560000e+02
  %55 = tail call float @llvm.fmuladd.f32(float %53, float %54, float %48)
  %56 = and i32 %51, 240
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %56) #10
  %58 = getelementptr inbounds float, float* %17, i64 4
  %59 = load float, float* %58, align 4, !tbaa !62
  %60 = tail call float @llvm.fmuladd.f32(float %57, float %59, float %55)
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 3
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !73
  %63 = zext i8 %62 to i32
  %64 = and i32 %63, 1
  %65 = tail call float @air.convert.f.f32.s.i32(i32 %64) #10
  %66 = fmul float %59, 2.560000e+02
  %67 = tail call float @llvm.fmuladd.f32(float %65, float %66, float %60)
  %68 = and i32 %63, 62
  %69 = tail call float @air.convert.f.f32.s.i32(i32 %68) #10
  %70 = getelementptr inbounds float, float* %17, i64 5
  %71 = load float, float* %70, align 4, !tbaa !62
  %72 = tail call float @llvm.fmuladd.f32(float %69, float %71, float %67)
  %73 = and i32 %63, 192
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %73) #10
  %75 = getelementptr inbounds float, float* %17, i64 6
  %76 = load float, float* %75, align 4, !tbaa !62
  %77 = tail call float @llvm.fmuladd.f32(float %74, float %76, float %72)
  %78 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 4
  %79 = load i8, i8 addrspace(1)* %78, align 1, !tbaa !73
  %80 = zext i8 %79 to i32
  %81 = and i32 %80, 7
  %82 = tail call float @air.convert.f.f32.s.i32(i32 %81) #10
  %83 = fmul float %76, 2.560000e+02
  %84 = tail call float @llvm.fmuladd.f32(float %82, float %83, float %77)
  %85 = and i32 %80, 248
  %86 = tail call float @air.convert.f.f32.s.i32(i32 %85) #10
  %87 = getelementptr inbounds float, float* %17, i64 7
  %88 = load float, float* %87, align 4, !tbaa !62
  %89 = tail call float @llvm.fmuladd.f32(float %86, float %88, float %84)
  br i1 %10, label %9, label %6, !llvm.loop !87
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap35mlx_qmv_f32xsum_v1_load_vector_safeIDF16bfLi8ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_i(bfloat addrspace(1)* noundef %0, float* noundef %1, i32 noundef %2) local_unnamed_addr #7 {
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
  %13 = load bfloat, bfloat addrspace(1)* %12, align 2, !tbaa !55
  %14 = fpext bfloat %13 to float
  %15 = or i32 %9, 1
  %16 = zext i32 %15 to i64
  %17 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %16
  %18 = load bfloat, bfloat addrspace(1)* %17, align 2, !tbaa !55
  %19 = fpext bfloat %18 to float
  %20 = fadd float %14, %19
  %21 = or i32 %9, 2
  %22 = zext i32 %21 to i64
  %23 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %22
  %24 = load bfloat, bfloat addrspace(1)* %23, align 2, !tbaa !55
  %25 = fpext bfloat %24 to float
  %26 = fadd float %20, %25
  %27 = or i32 %9, 3
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %28
  %30 = load bfloat, bfloat addrspace(1)* %29, align 2, !tbaa !55
  %31 = fpext bfloat %30 to float
  %32 = fadd float %26, %31
  %33 = or i32 %9, 4
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %34
  %36 = load bfloat, bfloat addrspace(1)* %35, align 2, !tbaa !55
  %37 = fpext bfloat %36 to float
  %38 = fadd float %32, %37
  %39 = or i32 %9, 5
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %40
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !55
  %43 = fpext bfloat %42 to float
  %44 = fadd float %38, %43
  %45 = or i32 %9, 6
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !55
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %9, 7
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !55
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = fadd float %10, %56
  %58 = getelementptr inbounds float, float* %1, i64 %11
  store float %14, float* %58, align 4, !tbaa !62
  %59 = fmul float %19, 3.125000e-02
  %60 = getelementptr inbounds float, float* %1, i64 %16
  store float %59, float* %60, align 4, !tbaa !62
  %61 = fmul float %25, 2.500000e-01
  %62 = getelementptr inbounds float, float* %1, i64 %22
  store float %61, float* %62, align 4, !tbaa !62
  %63 = fmul float %31, 7.812500e-03
  %64 = getelementptr inbounds float, float* %1, i64 %28
  store float %63, float* %64, align 4, !tbaa !62
  %65 = fmul float %37, 6.250000e-02
  %66 = getelementptr inbounds float, float* %1, i64 %34
  store float %65, float* %66, align 4, !tbaa !62
  %67 = fmul float %43, 5.000000e-01
  %68 = getelementptr inbounds float, float* %1, i64 %40
  store float %67, float* %68, align 4, !tbaa !62
  %69 = fmul float %49, 1.562500e-02
  %70 = getelementptr inbounds float, float* %1, i64 %46
  store float %69, float* %70, align 4, !tbaa !62
  %71 = fmul float %55, 1.250000e-01
  %72 = getelementptr inbounds float, float* %1, i64 %52
  store float %71, float* %72, align 4, !tbaa !62
  %73 = add nuw nsw i32 %9, 8
  %74 = icmp slt i32 %73, %2
  br i1 %74, label %8, label %5, !llvm.loop !88

75:                                               ; preds = %76, %5
  ret float %6

76:                                               ; preds = %76, %5
  %77 = phi i32 [ %80, %76 ], [ %2, %5 ]
  %78 = sext i32 %77 to i64
  %79 = getelementptr inbounds float, float* %1, i64 %78
  store float 0.000000e+00, float* %79, align 4, !tbaa !62
  %80 = add i32 %77, 1
  %81 = icmp eq i32 %80, 8
  br i1 %81, label %75, label %76, !llvm.loop !89
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #7 {
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
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !73
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 31
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #10
  %28 = load float, float* %20, align 4, !tbaa !62
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 224
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #10
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !62
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !73
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 3
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 124
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !62
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = and i32 %37, 128
  %48 = tail call float @air.convert.f.f32.s.i32(i32 %47) #10
  %49 = getelementptr inbounds float, float* %20, i64 3
  %50 = load float, float* %49, align 4, !tbaa !62
  %51 = tail call float @llvm.fmuladd.f32(float %48, float %50, float %46)
  %52 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %53 = load i8, i8 addrspace(1)* %52, align 1, !tbaa !73
  %54 = zext i8 %53 to i32
  %55 = and i32 %54, 15
  %56 = tail call float @air.convert.f.f32.s.i32(i32 %55) #10
  %57 = fmul float %50, 2.560000e+02
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %51)
  %59 = and i32 %54, 240
  %60 = tail call float @air.convert.f.f32.s.i32(i32 %59) #10
  %61 = getelementptr inbounds float, float* %20, i64 4
  %62 = load float, float* %61, align 4, !tbaa !62
  %63 = tail call float @llvm.fmuladd.f32(float %60, float %62, float %58)
  %64 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 3
  %65 = load i8, i8 addrspace(1)* %64, align 1, !tbaa !73
  %66 = zext i8 %65 to i32
  %67 = and i32 %66, 1
  %68 = tail call float @air.convert.f.f32.s.i32(i32 %67) #10
  %69 = fmul float %62, 2.560000e+02
  %70 = tail call float @llvm.fmuladd.f32(float %68, float %69, float %63)
  %71 = and i32 %66, 62
  %72 = tail call float @air.convert.f.f32.s.i32(i32 %71) #10
  %73 = getelementptr inbounds float, float* %20, i64 5
  %74 = load float, float* %73, align 4, !tbaa !62
  %75 = tail call float @llvm.fmuladd.f32(float %72, float %74, float %70)
  %76 = and i32 %66, 192
  %77 = tail call float @air.convert.f.f32.s.i32(i32 %76) #10
  %78 = getelementptr inbounds float, float* %20, i64 6
  %79 = load float, float* %78, align 4, !tbaa !62
  %80 = tail call float @llvm.fmuladd.f32(float %77, float %79, float %75)
  %81 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 4
  %82 = load i8, i8 addrspace(1)* %81, align 1, !tbaa !73
  %83 = zext i8 %82 to i32
  %84 = and i32 %83, 7
  %85 = tail call float @air.convert.f.f32.s.i32(i32 %84) #10
  %86 = fmul float %79, 2.560000e+02
  %87 = tail call float @llvm.fmuladd.f32(float %85, float %86, float %80)
  %88 = and i32 %83, 248
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #10
  %90 = getelementptr inbounds float, float* %20, i64 7
  %91 = load float, float* %90, align 4, !tbaa !62
  %92 = tail call float @llvm.fmuladd.f32(float %89, float %91, float %87)
  %93 = add nuw nsw i32 %14, 1
  %94 = icmp eq i32 %93, %7
  br i1 %94, label %9, label %13, !llvm.loop !90
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %23

20:                                               ; preds = %12
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !46
  br label %40

23:                                               ; preds = %12
  %24 = shl i32 %11, 4
  %25 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %26 = zext i32 %9 to i64
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %28 = load i64, i64 addrspace(2)* %27, align 8, !tbaa !53
  %29 = mul i64 %28, %26
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %31 = load i64, i64 addrspace(2)* %30, align 8, !tbaa !72
  %32 = mul i64 %31, %26
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !46
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %32
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %37 = load i64, i64 addrspace(2)* %36, align 8
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %39 = load i64, i64 addrspace(2)* %38, align 8
  br label %48

40:                                               ; preds = %153, %20
  %41 = phi i32 [ %22, %20 ], [ %34, %153 ]
  %42 = icmp eq i32 %11, 0
  %43 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %44 = zext i32 %41 to i64
  %45 = mul i64 %44, %10
  %46 = zext i32 %8 to i64
  %47 = add i64 %45, %46
  br label %157

48:                                               ; preds = %153, %23
  %49 = phi i32 [ 0, %23 ], [ %154, %153 ]
  %50 = add i32 %49, %24
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %51
  br label %53

53:                                               ; preds = %53, %48
  %54 = phi i1 [ true, %48 ], [ false, %53 ]
  %55 = phi i32 [ 0, %48 ], [ 8, %53 ]
  %56 = phi float [ 0.000000e+00, %48 ], [ %103, %53 ]
  %57 = zext i32 %55 to i64
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %57
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !55
  %60 = fpext bfloat %59 to float
  %61 = or i32 %55, 1
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %62
  %64 = load bfloat, bfloat addrspace(1)* %63, align 2, !tbaa !55
  %65 = fpext bfloat %64 to float
  %66 = fadd float %60, %65
  %67 = or i32 %55, 2
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %68
  %70 = load bfloat, bfloat addrspace(1)* %69, align 2, !tbaa !55
  %71 = fpext bfloat %70 to float
  %72 = fadd float %66, %71
  %73 = or i32 %55, 3
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %74
  %76 = load bfloat, bfloat addrspace(1)* %75, align 2, !tbaa !55
  %77 = fpext bfloat %76 to float
  %78 = fadd float %72, %77
  %79 = or i32 %55, 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %80
  %82 = load bfloat, bfloat addrspace(1)* %81, align 2, !tbaa !55
  %83 = fpext bfloat %82 to float
  %84 = fadd float %78, %83
  %85 = or i32 %55, 5
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %86
  %88 = load bfloat, bfloat addrspace(1)* %87, align 2, !tbaa !55
  %89 = fpext bfloat %88 to float
  %90 = fadd float %84, %89
  %91 = or i32 %55, 6
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %92
  %94 = load bfloat, bfloat addrspace(1)* %93, align 2, !tbaa !55
  %95 = fpext bfloat %94 to float
  %96 = fadd float %90, %95
  %97 = or i32 %55, 7
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %98
  %100 = load bfloat, bfloat addrspace(1)* %99, align 2, !tbaa !55
  %101 = fpext bfloat %100 to float
  %102 = fadd float %96, %101
  %103 = fadd float %56, %102
  %104 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %57
  store float %60, float* %104, align 4, !tbaa !62
  %105 = fmul float %65, 3.125000e-02
  %106 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %62
  store float %105, float* %106, align 4, !tbaa !62
  %107 = fmul float %71, 2.500000e-01
  %108 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %68
  store float %107, float* %108, align 4, !tbaa !62
  %109 = fmul float %77, 7.812500e-03
  %110 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %74
  store float %109, float* %110, align 4, !tbaa !62
  %111 = fmul float %83, 6.250000e-02
  %112 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %80
  store float %111, float* %112, align 4, !tbaa !62
  %113 = fmul float %89, 5.000000e-01
  %114 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %86
  store float %113, float* %114, align 4, !tbaa !62
  %115 = fmul float %95, 1.562500e-02
  %116 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %92
  store float %115, float* %116, align 4, !tbaa !62
  %117 = fmul float %101, 1.250000e-01
  %118 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %98
  store float %117, float* %118, align 4, !tbaa !62
  br i1 %54, label %53, label %119, !llvm.loop !80

119:                                              ; preds = %53
  %120 = lshr i32 %50, 7
  %121 = mul nuw nsw i64 %51, 5
  %122 = lshr exact i64 %121, 3
  %123 = zext i32 %120 to i64
  %124 = getelementptr inbounds i8, i8 addrspace(1)* %35, i64 %122
  br label %125

125:                                              ; preds = %150, %119
  %126 = phi i32 [ 0, %119 ], [ %151, %150 ]
  %127 = add i32 %126, %8
  %128 = icmp ult i32 %127, %34
  br i1 %128, label %129, label %150

129:                                              ; preds = %125
  %130 = zext i32 %127 to i64
  %131 = mul i64 %37, %130
  %132 = getelementptr inbounds i8, i8 addrspace(1)* %124, i64 %131
  %133 = mul i64 %39, %130
  %134 = add i64 %133, %29
  %135 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %134
  %136 = bitcast i8 addrspace(1)* %135 to bfloat addrspace(1)*
  %137 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %134
  %138 = bitcast i8 addrspace(1)* %137 to bfloat addrspace(1)*
  %139 = getelementptr inbounds bfloat, bfloat addrspace(1)* %136, i64 %123
  %140 = load bfloat, bfloat addrspace(1)* %139, align 2, !tbaa !55
  %141 = fpext bfloat %140 to float
  %142 = getelementptr inbounds bfloat, bfloat addrspace(1)* %138, i64 %123
  %143 = load bfloat, bfloat addrspace(1)* %142, align 2, !tbaa !55
  %144 = fpext bfloat %143 to float
  %145 = call fast float @_ZN22r5_raw_odd_control_tap23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %132, float* noundef nonnull %25, float noundef %141, float noundef %144, float noundef %103) #16
  %146 = zext i32 %126 to i64
  %147 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !62
  %149 = fadd float %145, %148
  store float %149, float* %147, align 4, !tbaa !62
  br label %150

150:                                              ; preds = %129, %125
  %151 = add nuw nsw i32 %126, 1
  %152 = icmp eq i32 %151, 4
  br i1 %152, label %153, label %125, !llvm.loop !91

153:                                              ; preds = %150
  %154 = add i32 %49, 512
  %155 = icmp ult i32 %154, %18
  br i1 %155, label %48, label %40, !llvm.loop !92

156:                                              ; preds = %182
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #13
  ret void

157:                                              ; preds = %182, %40
  %158 = phi i32 [ 0, %40 ], [ %183, %182 ]
  %159 = add i32 %158, %8
  %160 = icmp ult i32 %159, %41
  br i1 %160, label %161, label %182

161:                                              ; preds = %157
  %162 = zext i32 %158 to i64
  %163 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %162
  %164 = load float, float* %163, align 4, !tbaa !62
  %165 = call fast float @air.simd_sum.f32(float %164) #15
  br i1 %42, label %166, label %182

166:                                              ; preds = %161
  %167 = fptrunc float %165 to bfloat
  %168 = bitcast float %165 to i32
  %169 = and i32 %168, 2139095040
  %170 = icmp eq i32 %169, 2139095040
  br i1 %170, label %176, label %171

171:                                              ; preds = %166
  %172 = fpext bfloat %167 to float
  %173 = bitcast float %172 to i32
  %174 = and i32 %173, 2139095040
  %175 = icmp eq i32 %174, 2139095040
  br i1 %175, label %176, label %178

176:                                              ; preds = %171, %166
  %177 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %43, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %178

178:                                              ; preds = %176, %171
  %179 = add i64 %47, %162
  %180 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %179
  store bfloat %167, bfloat addrspace(1)* %180, align 2, !tbaa !55
  %181 = getelementptr inbounds float, float addrspace(1)* %6, i64 %179
  store float %165, float addrspace(1)* %181, align 4, !tbaa !62
  br label %182

182:                                              ; preds = %178, %161, %157
  %183 = add nuw nsw i32 %158, 1
  %184 = icmp eq i32 %183, 4
  br i1 %184, label %156, label %157, !llvm.loop !93
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt128ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [8 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp ugt i32 %18, 256
  %20 = shl i32 %11, 3
  br i1 %19, label %21, label %182

21:                                               ; preds = %12
  %22 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 2
  %24 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 4
  %25 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 6
  %26 = zext i32 %9 to i64
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %28 = load i64, i64 addrspace(2)* %27, align 8, !tbaa !53
  %29 = mul i64 %28, %26
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %31 = load i64, i64 addrspace(2)* %30, align 8, !tbaa !72
  %32 = mul i64 %31, %26
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !46
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %32
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %37 = load i64, i64 addrspace(2)* %36, align 8
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %39 = load i64, i64 addrspace(2)* %38, align 8
  br label %40

40:                                               ; preds = %171, %21
  %41 = phi i32 [ 0, %21 ], [ %172, %171 ]
  %42 = add i32 %41, %20
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !55
  %46 = fpext bfloat %45 to float
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 1
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !55
  %49 = fpext bfloat %48 to float
  %50 = fadd float %46, %49
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 2
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !55
  %53 = fpext bfloat %52 to float
  %54 = fadd float %50, %53
  %55 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 3
  %56 = load bfloat, bfloat addrspace(1)* %55, align 2, !tbaa !55
  %57 = fpext bfloat %56 to float
  %58 = fadd float %54, %57
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 4
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !55
  %61 = fpext bfloat %60 to float
  %62 = fadd float %58, %61
  %63 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 5
  %64 = load bfloat, bfloat addrspace(1)* %63, align 2, !tbaa !55
  %65 = fpext bfloat %64 to float
  %66 = fadd float %62, %65
  %67 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 6
  %68 = load bfloat, bfloat addrspace(1)* %67, align 2, !tbaa !55
  %69 = fpext bfloat %68 to float
  %70 = fadd float %66, %69
  %71 = getelementptr inbounds bfloat, bfloat addrspace(1)* %44, i64 7
  %72 = load bfloat, bfloat addrspace(1)* %71, align 2, !tbaa !55
  %73 = fpext bfloat %72 to float
  %74 = fadd float %70, %73
  %75 = fadd float %74, 0.000000e+00
  %76 = fmul float %49, 3.125000e-02
  %77 = fmul float %53, 2.500000e-01
  %78 = fmul float %57, 7.812500e-03
  %79 = fmul float %61, 6.250000e-02
  %80 = fmul float %65, 5.000000e-01
  %81 = fmul float %69, 1.562500e-02
  %82 = fmul float %73, 1.250000e-01
  %83 = lshr i32 %42, 7
  %84 = mul nuw nsw i64 %43, 5
  %85 = lshr exact i64 %84, 3
  %86 = zext i32 %83 to i64
  %87 = getelementptr inbounds i8, i8 addrspace(1)* %35, i64 %85
  %88 = fmul float %76, 2.560000e+02
  %89 = fmul float %78, 2.560000e+02
  %90 = fmul float %79, 2.560000e+02
  %91 = fmul float %81, 2.560000e+02
  br label %92

92:                                               ; preds = %168, %40
  %93 = phi i32 [ 0, %40 ], [ %169, %168 ]
  %94 = add i32 %93, %8
  %95 = icmp ult i32 %94, %34
  br i1 %95, label %96, label %168

96:                                               ; preds = %92
  %97 = zext i32 %94 to i64
  %98 = mul i64 %37, %97
  %99 = getelementptr inbounds i8, i8 addrspace(1)* %87, i64 %98
  %100 = mul i64 %39, %97
  %101 = add i64 %100, %29
  %102 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %101
  %103 = bitcast i8 addrspace(1)* %102 to bfloat addrspace(1)*
  %104 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %101
  %105 = bitcast i8 addrspace(1)* %104 to bfloat addrspace(1)*
  %106 = getelementptr inbounds bfloat, bfloat addrspace(1)* %103, i64 %86
  %107 = load bfloat, bfloat addrspace(1)* %106, align 2, !tbaa !55
  %108 = fpext bfloat %107 to float
  %109 = getelementptr inbounds bfloat, bfloat addrspace(1)* %105, i64 %86
  %110 = load bfloat, bfloat addrspace(1)* %109, align 2, !tbaa !55
  %111 = fpext bfloat %110 to float
  %112 = load i8, i8 addrspace(1)* %99, align 1, !tbaa !73
  %113 = zext i8 %112 to i32
  %114 = and i32 %113, 31
  %115 = tail call float @air.convert.f.f32.s.i32(i32 %114) #10
  %116 = and i32 %113, 224
  %117 = tail call float @air.convert.f.f32.s.i32(i32 %116) #10
  %118 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 1
  %119 = load i8, i8 addrspace(1)* %118, align 1, !tbaa !73
  %120 = zext i8 %119 to i32
  %121 = and i32 %120, 3
  %122 = tail call float @air.convert.f.f32.s.i32(i32 %121) #10
  %123 = and i32 %120, 124
  %124 = tail call float @air.convert.f.f32.s.i32(i32 %123) #10
  %125 = and i32 %120, 128
  %126 = tail call float @air.convert.f.f32.s.i32(i32 %125) #10
  %127 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 2
  %128 = load i8, i8 addrspace(1)* %127, align 1, !tbaa !73
  %129 = zext i8 %128 to i32
  %130 = and i32 %129, 15
  %131 = tail call float @air.convert.f.f32.s.i32(i32 %130) #10
  %132 = and i32 %129, 240
  %133 = tail call float @air.convert.f.f32.s.i32(i32 %132) #10
  %134 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 3
  %135 = load i8, i8 addrspace(1)* %134, align 1, !tbaa !73
  %136 = zext i8 %135 to i32
  %137 = and i32 %136, 1
  %138 = tail call float @air.convert.f.f32.s.i32(i32 %137) #10
  %139 = and i32 %136, 62
  %140 = tail call float @air.convert.f.f32.s.i32(i32 %139) #10
  %141 = and i32 %136, 192
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #10
  %143 = getelementptr inbounds i8, i8 addrspace(1)* %99, i64 4
  %144 = load i8, i8 addrspace(1)* %143, align 1, !tbaa !73
  %145 = zext i8 %144 to i32
  %146 = and i32 %145, 7
  %147 = tail call float @air.convert.f.f32.s.i32(i32 %146) #10
  %148 = and i32 %145, 248
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #10
  %150 = tail call float @llvm.fmuladd.f32(float %115, float %46, float 0.000000e+00) #13
  %151 = tail call float @llvm.fmuladd.f32(float %117, float %76, float %150) #13
  %152 = tail call float @llvm.fmuladd.f32(float %122, float %88, float %151) #13
  %153 = tail call float @llvm.fmuladd.f32(float %124, float %77, float %152) #13
  %154 = tail call float @llvm.fmuladd.f32(float %126, float %78, float %153) #13
  %155 = tail call float @llvm.fmuladd.f32(float %131, float %89, float %154) #13
  %156 = tail call float @llvm.fmuladd.f32(float %133, float %79, float %155) #13
  %157 = tail call float @llvm.fmuladd.f32(float %138, float %90, float %156) #13
  %158 = tail call float @llvm.fmuladd.f32(float %140, float %80, float %157) #13
  %159 = tail call float @llvm.fmuladd.f32(float %142, float %81, float %158) #13
  %160 = tail call float @llvm.fmuladd.f32(float %147, float %91, float %159) #13
  %161 = tail call float @llvm.fmuladd.f32(float %149, float %82, float %160) #13
  %162 = fmul float %75, %111
  %163 = tail call float @llvm.fmuladd.f32(float %108, float %161, float %162) #13
  %164 = zext i32 %93 to i64
  %165 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %164
  %166 = load float, float* %165, align 4, !tbaa !62
  %167 = fadd float %166, %163
  store float %167, float* %165, align 4, !tbaa !62
  br label %168

168:                                              ; preds = %96, %92
  %169 = add nuw nsw i32 %93, 1
  %170 = icmp eq i32 %169, 4
  br i1 %170, label %171, label %92, !llvm.loop !94

171:                                              ; preds = %168
  %172 = add i32 %41, 256
  %173 = icmp ugt i32 %18, %172
  %174 = sub i32 %18, %172
  %175 = icmp ugt i32 %174, 256
  %176 = and i1 %173, %175
  br i1 %176, label %40, label %177, !llvm.loop !95

177:                                              ; preds = %171
  %178 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 1
  %179 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 3
  %180 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 5
  %181 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 7
  store float %46, float* %22, align 4, !tbaa !62
  store float %76, float* %178, align 4, !tbaa !62
  store float %77, float* %23, align 4, !tbaa !62
  store float %78, float* %179, align 4, !tbaa !62
  store float %79, float* %24, align 4, !tbaa !62
  store float %80, float* %180, align 4, !tbaa !62
  store float %81, float* %25, align 4, !tbaa !62
  store float %82, float* %181, align 4, !tbaa !62
  br label %182

182:                                              ; preds = %177, %12
  %183 = phi i32 [ %172, %177 ], [ 0, %12 ]
  %184 = phi i32 [ %174, %177 ], [ %18, %12 ]
  %185 = icmp ugt i32 %184, %20
  br i1 %185, label %186, label %189

186:                                              ; preds = %182
  %187 = sub i32 %184, %20
  %188 = tail call i32 @air.min.u.i32(i32 %187, i32 8) #10
  br label %189

189:                                              ; preds = %186, %182
  %190 = phi i32 [ %188, %186 ], [ 0, %182 ]
  %191 = icmp eq i32 %190, 0
  br i1 %191, label %192, label %195

192:                                              ; preds = %189
  %193 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %194 = load i32, i32 addrspace(2)* %193, align 4, !tbaa !46
  br label %248

195:                                              ; preds = %189
  %196 = add i32 %183, %20
  %197 = zext i32 %196 to i64
  %198 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %197
  %199 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %200 = call fast float @_ZN22r5_raw_odd_control_tap35mlx_qmv_f32xsum_v1_load_vector_safeIDF16bfLi8ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_i(bfloat addrspace(1)* noundef %198, float* noundef nonnull %199, i32 noundef %190) #14
  %201 = zext i32 %9 to i64
  %202 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %203 = load i64, i64 addrspace(2)* %202, align 8, !tbaa !53
  %204 = mul i64 %203, %201
  %205 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %206 = load i64, i64 addrspace(2)* %205, align 8, !tbaa !72
  %207 = mul i64 %206, %201
  %208 = lshr i32 %196, 7
  %209 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %210 = load i32, i32 addrspace(2)* %209, align 4, !tbaa !46
  %211 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %207
  %212 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %213 = load i64, i64 addrspace(2)* %212, align 8
  %214 = mul nuw nsw i64 %197, 5
  %215 = lshr exact i64 %214, 3
  %216 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %217 = load i64, i64 addrspace(2)* %216, align 8
  %218 = zext i32 %208 to i64
  %219 = getelementptr inbounds i8, i8 addrspace(1)* %211, i64 %215
  br label %220

220:                                              ; preds = %245, %195
  %221 = phi i32 [ 0, %195 ], [ %246, %245 ]
  %222 = add i32 %221, %8
  %223 = icmp ult i32 %222, %210
  br i1 %223, label %224, label %245

224:                                              ; preds = %220
  %225 = zext i32 %222 to i64
  %226 = mul i64 %213, %225
  %227 = getelementptr inbounds i8, i8 addrspace(1)* %219, i64 %226
  %228 = mul i64 %217, %225
  %229 = add i64 %228, %204
  %230 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %229
  %231 = bitcast i8 addrspace(1)* %230 to bfloat addrspace(1)*
  %232 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %229
  %233 = bitcast i8 addrspace(1)* %232 to bfloat addrspace(1)*
  %234 = getelementptr inbounds bfloat, bfloat addrspace(1)* %231, i64 %218
  %235 = load bfloat, bfloat addrspace(1)* %234, align 2, !tbaa !55
  %236 = fpext bfloat %235 to float
  %237 = getelementptr inbounds bfloat, bfloat addrspace(1)* %233, i64 %218
  %238 = load bfloat, bfloat addrspace(1)* %237, align 2, !tbaa !55
  %239 = fpext bfloat %238 to float
  %240 = call fast float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %227, float* noundef nonnull %199, float noundef %236, float noundef %239, float noundef %200, i32 noundef %190) #16
  %241 = zext i32 %221 to i64
  %242 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %241
  %243 = load float, float* %242, align 4, !tbaa !62
  %244 = fadd float %240, %243
  store float %244, float* %242, align 4, !tbaa !62
  br label %245

245:                                              ; preds = %224, %220
  %246 = add nuw nsw i32 %221, 1
  %247 = icmp eq i32 %246, 4
  br i1 %247, label %248, label %220, !llvm.loop !94

248:                                              ; preds = %245, %192
  %249 = phi i32 [ %194, %192 ], [ %210, %245 ]
  %250 = icmp eq i32 %11, 0
  %251 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %252 = zext i32 %249 to i64
  %253 = mul i64 %252, %10
  %254 = zext i32 %8 to i64
  %255 = add i64 %253, %254
  br label %257

256:                                              ; preds = %282
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #13
  ret void

257:                                              ; preds = %282, %248
  %258 = phi i32 [ 0, %248 ], [ %283, %282 ]
  %259 = add i32 %258, %8
  %260 = icmp ult i32 %259, %249
  br i1 %260, label %261, label %282

261:                                              ; preds = %257
  %262 = zext i32 %258 to i64
  %263 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %262
  %264 = load float, float* %263, align 4, !tbaa !62
  %265 = call fast float @air.simd_sum.f32(float %264) #15
  br i1 %250, label %266, label %282

266:                                              ; preds = %261
  %267 = fptrunc float %265 to bfloat
  %268 = bitcast float %265 to i32
  %269 = and i32 %268, 2139095040
  %270 = icmp eq i32 %269, 2139095040
  br i1 %270, label %276, label %271

271:                                              ; preds = %266
  %272 = fpext bfloat %267 to float
  %273 = bitcast float %272 to i32
  %274 = and i32 %273, 2139095040
  %275 = icmp eq i32 %274, 2139095040
  br i1 %275, label %276, label %278

276:                                              ; preds = %271, %266
  %277 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %251, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %278

278:                                              ; preds = %276, %271
  %279 = add i64 %255, %262
  %280 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %279
  store bfloat %267, bfloat addrspace(1)* %280, align 2, !tbaa !55
  %281 = getelementptr inbounds float, float addrspace(1)* %6, i64 %279
  store float %265, float addrspace(1)* %281, align 4, !tbaa !62
  br label %282

282:                                              ; preds = %278, %261, %257
  %283 = add nuw nsw i32 %258, 1
  %284 = icmp eq i32 %283, 4
  br i1 %284, label %256, label %257, !llvm.loop !96
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [8 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %24, label %20

20:                                               ; preds = %12
  %21 = shl i32 %11, 3
  %22 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  br label %33

24:                                               ; preds = %72, %12
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !46
  %27 = icmp eq i32 %11, 0
  %28 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %29 = zext i32 %26 to i64
  %30 = mul i64 %29, %10
  %31 = zext i32 %8 to i64
  %32 = add i64 %30, %31
  br label %76

33:                                               ; preds = %72, %20
  %34 = phi i32 [ 0, %20 ], [ %73, %72 ]
  %35 = add i32 %34, %21
  %36 = zext i32 %35 to i64
  %37 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %36
  br label %38

38:                                               ; preds = %38, %33
  %39 = phi i1 [ true, %33 ], [ false, %38 ]
  %40 = phi i32 [ 0, %33 ], [ 4, %38 ]
  %41 = phi float [ 0.000000e+00, %33 ], [ %64, %38 ]
  %42 = zext i32 %40 to i64
  %43 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %42
  %44 = load bfloat, bfloat addrspace(1)* %43, align 2, !tbaa !55
  %45 = fpext bfloat %44 to float
  %46 = or i32 %40, 1
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %47
  %49 = load bfloat, bfloat addrspace(1)* %48, align 2, !tbaa !55
  %50 = fpext bfloat %49 to float
  %51 = fadd float %45, %50
  %52 = or i32 %40, 2
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %53
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !55
  %56 = fpext bfloat %55 to float
  %57 = fadd float %51, %56
  %58 = or i32 %40, 3
  %59 = zext i32 %58 to i64
  %60 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %59
  %61 = load bfloat, bfloat addrspace(1)* %60, align 2, !tbaa !55
  %62 = fpext bfloat %61 to float
  %63 = fadd float %57, %62
  %64 = fadd float %41, %63
  %65 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %42
  store float %45, float* %65, align 4, !tbaa !62
  %66 = fmul float %50, 1.562500e-02
  %67 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %47
  store float %66, float* %67, align 4, !tbaa !62
  %68 = fmul float %56, 6.250000e-02
  %69 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %53
  store float %68, float* %69, align 4, !tbaa !62
  %70 = fmul float %62, 2.500000e-01
  %71 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %59
  store float %70, float* %71, align 4, !tbaa !62
  br i1 %39, label %38, label %72, !llvm.loop !97

72:                                               ; preds = %38
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %35, i32 noundef %9, float* noundef nonnull %22, float noundef %64, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %23) #12
  %73 = add i32 %34, 256
  %74 = icmp ult i32 %73, %18
  br i1 %74, label %33, label %24, !llvm.loop !98

75:                                               ; preds = %101
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #13
  ret void

76:                                               ; preds = %101, %24
  %77 = phi i32 [ 0, %24 ], [ %102, %101 ]
  %78 = add i32 %77, %8
  %79 = icmp ult i32 %78, %26
  br i1 %79, label %80, label %101

80:                                               ; preds = %76
  %81 = zext i32 %77 to i64
  %82 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %81
  %83 = load float, float* %82, align 4, !tbaa !62
  %84 = call fast float @air.simd_sum.f32(float %83) #15
  br i1 %27, label %85, label %101

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
  %96 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %28, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %97

97:                                               ; preds = %95, %90
  %98 = add i64 %32, %81
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %98
  store bfloat %86, bfloat addrspace(1)* %99, align 2, !tbaa !55
  %100 = getelementptr inbounds float, float addrspace(1)* %6, i64 %98
  store float %84, float addrspace(1)* %100, align 4, !tbaa !62
  br label %101

101:                                              ; preds = %97, %80, %76
  %102 = add nuw nsw i32 %77, 1
  %103 = icmp eq i32 %102, 4
  br i1 %103, label %75, label %76, !llvm.loop !99
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt6ELt64ELb0EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #5 {
  %13 = alloca [4 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #13
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !47
  %19 = icmp ugt i32 %18, 128
  %20 = shl i32 %11, 2
  br i1 %19, label %21, label %55

21:                                               ; preds = %12
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 1
  %24 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 2
  %25 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 3
  %26 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  br label %27

27:                                               ; preds = %27, %21
  %28 = phi i32 [ 0, %21 ], [ %50, %27 ]
  %29 = add i32 %28, %20
  %30 = zext i32 %29 to i64
  %31 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %30
  %32 = load bfloat, bfloat addrspace(1)* %31, align 2, !tbaa !55
  %33 = fpext bfloat %32 to float
  %34 = getelementptr inbounds bfloat, bfloat addrspace(1)* %31, i64 1
  %35 = load bfloat, bfloat addrspace(1)* %34, align 2, !tbaa !55
  %36 = fpext bfloat %35 to float
  %37 = fadd float %33, %36
  %38 = getelementptr inbounds bfloat, bfloat addrspace(1)* %31, i64 2
  %39 = load bfloat, bfloat addrspace(1)* %38, align 2, !tbaa !55
  %40 = fpext bfloat %39 to float
  %41 = fadd float %37, %40
  %42 = getelementptr inbounds bfloat, bfloat addrspace(1)* %31, i64 3
  %43 = load bfloat, bfloat addrspace(1)* %42, align 2, !tbaa !55
  %44 = fpext bfloat %43 to float
  %45 = fadd float %41, %44
  %46 = fadd float %45, 0.000000e+00
  store float %33, float* %22, align 4, !tbaa !62
  %47 = fmul float %36, 1.562500e-02
  store float %47, float* %23, align 4, !tbaa !62
  %48 = fmul float %40, 6.250000e-02
  store float %48, float* %24, align 4, !tbaa !62
  %49 = fmul float %44, 2.500000e-01
  store float %49, float* %25, align 4, !tbaa !62
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt6ELt64ELt4EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %29, i32 noundef %9, float* noundef nonnull %22, float noundef %46, i32 noundef 4, i1 noundef zeroext false, float* noundef nonnull %26) #12
  %50 = add i32 %28, 128
  %51 = icmp ugt i32 %18, %50
  %52 = sub i32 %18, %50
  %53 = icmp ugt i32 %52, 128
  %54 = and i1 %51, %53
  br i1 %54, label %27, label %55, !llvm.loop !100

55:                                               ; preds = %27, %12
  %56 = phi i32 [ 0, %12 ], [ %50, %27 ]
  %57 = phi i32 [ %18, %12 ], [ %52, %27 ]
  %58 = icmp ugt i32 %57, %20
  br i1 %58, label %59, label %62

59:                                               ; preds = %55
  %60 = sub i32 %57, %20
  %61 = call i32 @air.min.u.i32(i32 %60, i32 4) #10
  br label %62

62:                                               ; preds = %59, %55
  %63 = phi i32 [ %61, %59 ], [ 0, %55 ]
  %64 = icmp eq i32 %63, 0
  br i1 %64, label %65, label %68

65:                                               ; preds = %62
  %66 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %67 = load i32, i32 addrspace(2)* %66, align 4, !tbaa !46
  br label %166

68:                                               ; preds = %62
  %69 = add i32 %56, %20
  %70 = zext i32 %69 to i64
  %71 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %70
  %72 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  %73 = icmp sgt i32 %63, 0
  br i1 %73, label %77, label %74

74:                                               ; preds = %77, %68
  %75 = phi float [ 0.000000e+00, %68 ], [ %102, %77 ]
  %76 = icmp slt i32 %63, 4
  br i1 %76, label %112, label %118

77:                                               ; preds = %77, %68
  %78 = phi i32 [ %110, %77 ], [ 0, %68 ]
  %79 = phi float [ %102, %77 ], [ 0.000000e+00, %68 ]
  %80 = zext i32 %78 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %71, i64 %80
  %82 = load bfloat, bfloat addrspace(1)* %81, align 2, !tbaa !55
  %83 = fpext bfloat %82 to float
  %84 = or i32 %78, 1
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds bfloat, bfloat addrspace(1)* %71, i64 %85
  %87 = load bfloat, bfloat addrspace(1)* %86, align 2, !tbaa !55
  %88 = fpext bfloat %87 to float
  %89 = fadd float %83, %88
  %90 = or i32 %78, 2
  %91 = zext i32 %90 to i64
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %71, i64 %91
  %93 = load bfloat, bfloat addrspace(1)* %92, align 2, !tbaa !55
  %94 = fpext bfloat %93 to float
  %95 = fadd float %89, %94
  %96 = or i32 %78, 3
  %97 = zext i32 %96 to i64
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %71, i64 %97
  %99 = load bfloat, bfloat addrspace(1)* %98, align 2, !tbaa !55
  %100 = fpext bfloat %99 to float
  %101 = fadd float %95, %100
  %102 = fadd float %79, %101
  %103 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %80
  store float %83, float* %103, align 4, !tbaa !62
  %104 = fmul float %88, 1.562500e-02
  %105 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %85
  store float %104, float* %105, align 4, !tbaa !62
  %106 = fmul float %94, 6.250000e-02
  %107 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %91
  store float %106, float* %107, align 4, !tbaa !62
  %108 = fmul float %100, 2.500000e-01
  %109 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %97
  store float %108, float* %109, align 4, !tbaa !62
  %110 = add nuw nsw i32 %78, 4
  %111 = icmp slt i32 %110, %63
  br i1 %111, label %77, label %74, !llvm.loop !101

112:                                              ; preds = %112, %74
  %113 = phi i32 [ %116, %112 ], [ %63, %74 ]
  %114 = sext i32 %113 to i64
  %115 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %114
  store float 0.000000e+00, float* %115, align 4, !tbaa !62
  %116 = add i32 %113, 1
  %117 = icmp eq i32 %116, 4
  br i1 %117, label %118, label %112, !llvm.loop !102

118:                                              ; preds = %112, %74
  %119 = zext i32 %9 to i64
  %120 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %121 = load i64, i64 addrspace(2)* %120, align 8, !tbaa !53
  %122 = mul i64 %121, %119
  %123 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %124 = load i64, i64 addrspace(2)* %123, align 8, !tbaa !72
  %125 = mul i64 %124, %119
  %126 = lshr i32 %69, 6
  %127 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %128 = load i32, i32 addrspace(2)* %127, align 4, !tbaa !46
  %129 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %125
  %130 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %131 = load i64, i64 addrspace(2)* %130, align 8
  %132 = mul nuw nsw i64 %70, 6
  %133 = lshr exact i64 %132, 3
  %134 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %135 = load i64, i64 addrspace(2)* %134, align 8
  %136 = zext i32 %126 to i64
  %137 = getelementptr inbounds i8, i8 addrspace(1)* %129, i64 %133
  br label %138

138:                                              ; preds = %163, %118
  %139 = phi i32 [ 0, %118 ], [ %164, %163 ]
  %140 = add i32 %139, %8
  %141 = icmp ult i32 %140, %128
  br i1 %141, label %142, label %163

142:                                              ; preds = %138
  %143 = zext i32 %140 to i64
  %144 = mul i64 %131, %143
  %145 = getelementptr inbounds i8, i8 addrspace(1)* %137, i64 %144
  %146 = mul i64 %135, %143
  %147 = add i64 %146, %122
  %148 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %147
  %149 = bitcast i8 addrspace(1)* %148 to bfloat addrspace(1)*
  %150 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %147
  %151 = bitcast i8 addrspace(1)* %150 to bfloat addrspace(1)*
  %152 = getelementptr inbounds bfloat, bfloat addrspace(1)* %149, i64 %136
  %153 = load bfloat, bfloat addrspace(1)* %152, align 2, !tbaa !55
  %154 = fpext bfloat %153 to float
  %155 = getelementptr inbounds bfloat, bfloat addrspace(1)* %151, i64 %136
  %156 = load bfloat, bfloat addrspace(1)* %155, align 2, !tbaa !55
  %157 = fpext bfloat %156 to float
  %158 = call fast float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %145, float* noundef nonnull %72, float noundef %154, float noundef %157, float noundef %75, i32 noundef %63) #16
  %159 = zext i32 %139 to i64
  %160 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %159
  %161 = load float, float* %160, align 4, !tbaa !62
  %162 = fadd float %158, %161
  store float %162, float* %160, align 4, !tbaa !62
  br label %163

163:                                              ; preds = %142, %138
  %164 = add nuw nsw i32 %139, 1
  %165 = icmp eq i32 %164, 4
  br i1 %165, label %166, label %138, !llvm.loop !103

166:                                              ; preds = %163, %65
  %167 = phi i32 [ %67, %65 ], [ %128, %163 ]
  %168 = icmp eq i32 %11, 0
  %169 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %170 = zext i32 %167 to i64
  %171 = mul i64 %170, %10
  %172 = zext i32 %8 to i64
  %173 = add i64 %171, %172
  br label %175

174:                                              ; preds = %200
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #13
  ret void

175:                                              ; preds = %200, %166
  %176 = phi i32 [ 0, %166 ], [ %201, %200 ]
  %177 = add i32 %176, %8
  %178 = icmp ult i32 %177, %167
  br i1 %178, label %179, label %200

179:                                              ; preds = %175
  %180 = zext i32 %176 to i64
  %181 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %180
  %182 = load float, float* %181, align 4, !tbaa !62
  %183 = call fast float @air.simd_sum.f32(float %182) #15
  br i1 %168, label %184, label %200

184:                                              ; preds = %179
  %185 = fptrunc float %183 to bfloat
  %186 = bitcast float %183 to i32
  %187 = and i32 %186, 2139095040
  %188 = icmp eq i32 %187, 2139095040
  br i1 %188, label %194, label %189

189:                                              ; preds = %184
  %190 = fpext bfloat %185 to float
  %191 = bitcast float %190 to i32
  %192 = and i32 %191, 2139095040
  %193 = icmp eq i32 %192, 2139095040
  br i1 %193, label %194, label %196

194:                                              ; preds = %189, %184
  %195 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %169, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %196

196:                                              ; preds = %194, %189
  %197 = add i64 %173, %180
  %198 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %197
  store bfloat %185, bfloat addrspace(1)* %198, align 2, !tbaa !55
  %199 = getelementptr inbounds float, float addrspace(1)* %6, i64 %197
  store float %183, float addrspace(1)* %199, align 4, !tbaa !62
  br label %200

200:                                              ; preds = %196, %179, %175
  %201 = add nuw nsw i32 %176, 1
  %202 = icmp eq i32 %201, 4
  br i1 %202, label %174, label %175, !llvm.loop !104
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #5 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !53
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !72
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !46
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
  %49 = load bfloat, bfloat addrspace(1)* %48, align 2, !tbaa !55
  %50 = fpext bfloat %49 to float
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %47, i64 %31
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !55
  %53 = fpext bfloat %52 to float
  br i1 %10, label %54, label %60

54:                                               ; preds = %38
  %55 = tail call fast float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %41, float* noundef %7, float noundef %50, float noundef %53, float noundef %8, i32 noundef %9) #14
  %56 = zext i32 %35 to i64
  %57 = getelementptr inbounds float, float* %11, i64 %56
  %58 = load float, float* %57, align 4, !tbaa !62
  %59 = fadd float %55, %58
  store float %59, float* %57, align 4, !tbaa !62
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
  %72 = load i8, i8 addrspace(1)* %71, align 1, !tbaa !73
  %73 = zext i8 %72 to i32
  %74 = and i32 %73, 63
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = load float, float* %68, align 4, !tbaa !62
  %77 = tail call float @llvm.fmuladd.f32(float %75, float %76, float %63) #13
  %78 = and i32 %73, 192
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #10
  %80 = getelementptr inbounds float, float* %68, i64 1
  %81 = load float, float* %80, align 4, !tbaa !62
  %82 = tail call float @llvm.fmuladd.f32(float %79, float %81, float %77) #13
  %83 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 1
  %84 = load i8, i8 addrspace(1)* %83, align 1, !tbaa !73
  %85 = zext i8 %84 to i32
  %86 = and i32 %85, 15
  %87 = tail call float @air.convert.f.f32.s.i32(i32 %86) #10
  %88 = fmul float %81, 2.560000e+02
  %89 = tail call float @llvm.fmuladd.f32(float %87, float %88, float %82) #13
  %90 = and i32 %85, 240
  %91 = tail call float @air.convert.f.f32.s.i32(i32 %90) #10
  %92 = getelementptr inbounds float, float* %68, i64 2
  %93 = load float, float* %92, align 4, !tbaa !62
  %94 = tail call float @llvm.fmuladd.f32(float %91, float %93, float %89) #13
  %95 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 2
  %96 = load i8, i8 addrspace(1)* %95, align 1, !tbaa !73
  %97 = zext i8 %96 to i32
  %98 = and i32 %97, 3
  %99 = tail call float @air.convert.f.f32.s.i32(i32 %98) #10
  %100 = fmul float %93, 2.560000e+02
  %101 = tail call float @llvm.fmuladd.f32(float %99, float %100, float %94) #13
  %102 = and i32 %97, 252
  %103 = tail call float @air.convert.f.f32.s.i32(i32 %102) #10
  %104 = getelementptr inbounds float, float* %68, i64 3
  %105 = load float, float* %104, align 4, !tbaa !62
  %106 = tail call float @llvm.fmuladd.f32(float %103, float %105, float %101) #13
  br i1 %61, label %60, label %107, !llvm.loop !105

107:                                              ; preds = %60
  %108 = fmul float %53, %8
  %109 = tail call float @llvm.fmuladd.f32(float %50, float %106, float %108) #13
  %110 = zext i32 %35 to i64
  %111 = getelementptr inbounds float, float* %11, i64 %110
  %112 = load float, float* %111, align 4, !tbaa !62
  %113 = fadd float %112, %109
  store float %113, float* %111, align 4, !tbaa !62
  br label %114

114:                                              ; preds = %107, %54, %34
  %115 = add nuw nsw i32 %35, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %33, label %34, !llvm.loop !106
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #7 {
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
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !73
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 63
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #10
  %28 = load float, float* %20, align 4, !tbaa !62
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 192
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #10
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !62
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !73
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 15
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 240
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !62
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %48 = load i8, i8 addrspace(1)* %47, align 1, !tbaa !73
  %49 = zext i8 %48 to i32
  %50 = and i32 %49, 3
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %50) #10
  %52 = fmul float %45, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %46)
  %54 = and i32 %49, 252
  %55 = tail call float @air.convert.f.f32.s.i32(i32 %54) #10
  %56 = getelementptr inbounds float, float* %20, i64 3
  %57 = load float, float* %56, align 4, !tbaa !62
  %58 = tail call float @llvm.fmuladd.f32(float %55, float %57, float %53)
  %59 = add nuw nsw i32 %14, 1
  %60 = icmp eq i32 %59, %7
  br i1 %60, label %9, label %13, !llvm.loop !107
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt6ELt64ELt4EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #5 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !53
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !72
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !46
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
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !55
  %53 = fpext bfloat %52 to float
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %50, i64 %31
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !55
  %56 = fpext bfloat %55 to float
  br i1 %10, label %57, label %63

57:                                               ; preds = %41
  %58 = tail call fast float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %44, float* noundef %7, float noundef %53, float noundef %56, float noundef %8, i32 noundef %9) #14
  %59 = zext i32 %38 to i64
  %60 = getelementptr inbounds float, float* %11, i64 %59
  %61 = load float, float* %60, align 4, !tbaa !62
  %62 = fadd float %58, %61
  store float %62, float* %60, align 4, !tbaa !62
  br label %102

63:                                               ; preds = %41
  %64 = load i8, i8 addrspace(1)* %44, align 1, !tbaa !73
  %65 = zext i8 %64 to i32
  %66 = and i32 %65, 63
  %67 = tail call float @air.convert.f.f32.s.i32(i32 %66) #10
  %68 = load float, float* %7, align 4, !tbaa !62
  %69 = and i32 %65, 192
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %69) #10
  %71 = load float, float* %32, align 4, !tbaa !62
  %72 = getelementptr inbounds i8, i8 addrspace(1)* %44, i64 1
  %73 = load i8, i8 addrspace(1)* %72, align 1, !tbaa !73
  %74 = zext i8 %73 to i32
  %75 = and i32 %74, 15
  %76 = tail call float @air.convert.f.f32.s.i32(i32 %75) #10
  %77 = fmul float %71, 2.560000e+02
  %78 = and i32 %74, 240
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #10
  %80 = load float, float* %33, align 4, !tbaa !62
  %81 = getelementptr inbounds i8, i8 addrspace(1)* %44, i64 2
  %82 = load i8, i8 addrspace(1)* %81, align 1, !tbaa !73
  %83 = zext i8 %82 to i32
  %84 = and i32 %83, 3
  %85 = tail call float @air.convert.f.f32.s.i32(i32 %84) #10
  %86 = fmul float %80, 2.560000e+02
  %87 = and i32 %83, 252
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %87) #10
  %89 = load float, float* %34, align 4, !tbaa !62
  %90 = tail call float @llvm.fmuladd.f32(float %67, float %68, float 0.000000e+00) #13
  %91 = tail call float @llvm.fmuladd.f32(float %70, float %71, float %90) #13
  %92 = tail call float @llvm.fmuladd.f32(float %76, float %77, float %91) #13
  %93 = tail call float @llvm.fmuladd.f32(float %79, float %80, float %92) #13
  %94 = tail call float @llvm.fmuladd.f32(float %85, float %86, float %93) #13
  %95 = tail call float @llvm.fmuladd.f32(float %88, float %89, float %94) #13
  %96 = fmul float %56, %8
  %97 = tail call float @llvm.fmuladd.f32(float %53, float %95, float %96) #13
  %98 = zext i32 %38 to i64
  %99 = getelementptr inbounds float, float* %11, i64 %98
  %100 = load float, float* %99, align 4, !tbaa !62
  %101 = fadd float %100, %97
  store float %101, float* %99, align 4, !tbaa !62
  br label %102

102:                                              ; preds = %63, %57, %37
  %103 = add nuw nsw i32 %38, 1
  %104 = icmp eq i32 %103, 4
  br i1 %104, label %36, label %37, !llvm.loop !103
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi4ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #7 {
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
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !73
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 63
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #10
  %28 = load float, float* %20, align 4, !tbaa !62
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 192
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #10
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !62
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !73
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 15
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 240
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !62
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %48 = load i8, i8 addrspace(1)* %47, align 1, !tbaa !73
  %49 = zext i8 %48 to i32
  %50 = and i32 %49, 3
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %50) #10
  %52 = fmul float %45, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %46)
  %54 = and i32 %49, 252
  %55 = tail call float @air.convert.f.f32.s.i32(i32 %54) #10
  %56 = getelementptr inbounds float, float* %20, i64 3
  %57 = load float, float* %56, align 4, !tbaa !62
  %58 = tail call float @llvm.fmuladd.f32(float %55, float %57, float %53)
  %59 = add nuw nsw i32 %14, 1
  %60 = icmp eq i32 %59, %7
  br i1 %60, label %9, label %13, !llvm.loop !108
}

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #1 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #2 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #3 = { mustprogress nounwind willreturn }
attributes #4 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #5 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #6 = { argmemonly nofree nounwind willreturn writeonly }
attributes #7 = { inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #8 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #9 = { convergent mustprogress nounwind willreturn }
attributes #10 = { nounwind readnone willreturn }
attributes #11 = { nounwind willreturn }
attributes #12 = { convergent nobuiltin "no-builtins" }
attributes #13 = { nounwind }
attributes #14 = { nobuiltin "no-builtins" }
attributes #15 = { convergent nounwind willreturn }
attributes #16 = { nobuiltin nounwind "no-builtins" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9, !28, !29, !30}
!air.compile_options = !{!31, !32, !33}
!llvm.ident = !{!34}
!air.version = !{!35}
!air.language_version = !{!36}
!air.source_file_name = !{!37}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_control_probe_q4_g64, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !20, !22, !23, !24, !25, !26, !27}
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
!22 = !{i32 8, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"raw_f32"}
!23 = !{i32 9, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"group"}
!24 = !{i32 10, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads"}
!25 = !{i32 11, !"air.threads_per_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_width"}
!26 = !{i32 12, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
!27 = !{i32 13, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!28 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_control_probe_q5_g64, !10, !11}
!29 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_control_probe_q5_g128, !10, !11}
!30 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_control_probe_q6_g64, !10, !11}
!31 = !{!"air.compile.denorms_disable"}
!32 = !{!"air.compile.fast_math_enable"}
!33 = !{!"air.compile.framebuffer_fetch_enable"}
!34 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!35 = !{i32 2, i32 9, i32 0}
!36 = !{!"Metal", i32 4, i32 1, i32 0}
!37 = !{!"/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/control_probe.metal"}
!38 = !{!39, !40, i64 0}
!39 = !{!"_ZTS17FlashAffineParams", !40, i64 0, !40, i64 4, !40, i64 8, !40, i64 12, !40, i64 16, !40, i64 20, !40, i64 24, !40, i64 28, !43, i64 32, !43, i64 40, !43, i64 48, !43, i64 56}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C++ TBAA"}
!43 = !{!"long", !41, i64 0}
!44 = !{!39, !40, i64 4}
!45 = !{!39, !40, i64 16}
!46 = !{!39, !40, i64 12}
!47 = !{!39, !40, i64 8}
!48 = !{!39, !40, i64 20}
!49 = !{!39, !40, i64 24}
!50 = !{!39, !40, i64 28}
!51 = !{!39, !43, i64 32}
!52 = !{!39, !43, i64 48}
!53 = !{!39, !43, i64 56}
!54 = !{!43, !43, i64 0}
!55 = !{!56, !56, i64 0}
!56 = !{!"bfloat", !41, i64 0}
!57 = distinct !{!57, !58}
!58 = !{!"llvm.loop.mustprogress"}
!59 = distinct !{!59, !58}
!60 = distinct !{!60, !58}
!61 = distinct !{!61, !58}
!62 = !{!63, !63, i64 0}
!63 = !{!"float", !41, i64 0}
!64 = distinct !{!64, !58}
!65 = distinct !{!65, !58}
!66 = distinct !{!66, !58}
!67 = distinct !{!67, !58}
!68 = distinct !{!68, !58}
!69 = distinct !{!69, !58}
!70 = distinct !{!70, !58}
!71 = distinct !{!71, !58}
!72 = !{!39, !43, i64 40}
!73 = !{!41, !41, i64 0}
!74 = distinct !{!74, !58}
!75 = distinct !{!75, !58}
!76 = distinct !{!76, !58}
!77 = distinct !{!77, !58}
!78 = distinct !{!78, !58}
!79 = distinct !{!79, !58}
!80 = distinct !{!80, !58}
!81 = distinct !{!81, !58}
!82 = distinct !{!82, !58}
!83 = distinct !{!83, !58}
!84 = distinct !{!84, !58}
!85 = distinct !{!85, !58}
!86 = distinct !{!86, !58}
!87 = distinct !{!87, !58}
!88 = distinct !{!88, !58}
!89 = distinct !{!89, !58}
!90 = distinct !{!90, !58}
!91 = distinct !{!91, !58}
!92 = distinct !{!92, !58}
!93 = distinct !{!93, !58}
!94 = distinct !{!94, !58}
!95 = distinct !{!95, !58}
!96 = distinct !{!96, !58}
!97 = distinct !{!97, !58}
!98 = distinct !{!98, !58}
!99 = distinct !{!99, !58}
!100 = distinct !{!100, !58}
!101 = distinct !{!101, !58}
!102 = distinct !{!102, !58}
!103 = distinct !{!103, !58}
!104 = distinct !{!104, !58}
!105 = distinct !{!105, !58}
!106 = distinct !{!106, !58}
!107 = distinct !{!107, !58}
!108 = distinct !{!108, !58}
