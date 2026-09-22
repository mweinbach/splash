; ModuleID = '/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_guard_specialized_rowpair_sep22/kernel/_cpu_build_v1/candidate_probe.air'
source_filename = "/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_guard_specialized_rowpair_sep22/kernel/candidate_probe.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

%"struct.metal::_atomic" = type { i32 }
%struct.FlashAffineParams = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64 }

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_candidate_probe_q4_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
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
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca [4 x float], align 4
  %16 = alloca float, align 4
  %17 = alloca float, align 4
  %18 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %19 = load i32, i32 addrspace(2)* %18, align 8, !tbaa !38
  switch i32 %19, label %85 [
    i32 2560, label %28
    i32 6144, label %20
  ]

20:                                               ; preds = %11
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !44
  %23 = icmp eq i32 %22, 2560
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %25 = load i32, i32 addrspace(2)* %24, align 8
  %26 = icmp eq i32 %25, 5
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %35, label %85

28:                                               ; preds = %11
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !44
  switch i32 %30, label %85 [
    i32 10240, label %31
    i32 12288, label %31
  ]

31:                                               ; preds = %28, %28
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %33 = load i32, i32 addrspace(2)* %32, align 8, !tbaa !45
  %34 = icmp eq i32 %33, 5
  br i1 %34, label %35, label %85

35:                                               ; preds = %31, %20
  %36 = phi i32 [ 2560, %20 ], [ %30, %31 ]
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !46
  %39 = icmp eq i32 %38, 1
  br i1 %39, label %40, label %85

40:                                               ; preds = %35
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %42 = load i32, i32 addrspace(2)* %41, align 8, !tbaa !47
  %43 = icmp eq i32 %42, 1
  br i1 %43, label %44, label %85

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %46 = load i32, i32 addrspace(2)* %45, align 4, !tbaa !48
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %48, label %85

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %50 = load i32, i32 addrspace(2)* %49, align 4, !tbaa !49
  %51 = icmp eq i32 %50, 4
  br i1 %51, label %52, label %85

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %54 = load i32, i32 addrspace(2)* %53, align 8, !tbaa !50
  %55 = icmp eq i32 %54, 64
  %56 = and i32 %19, 511
  %57 = icmp eq i32 %56, 0
  %58 = select i1 %55, i1 %57, i1 false
  %59 = and i32 %36, 7
  %60 = icmp eq i32 %59, 0
  %61 = select i1 %58, i1 %60, i1 false
  br i1 %61, label %62, label %85

62:                                               ; preds = %52
  %63 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %64 = load i64, i64 addrspace(2)* %63, align 8, !tbaa !51
  %65 = lshr i32 %19, 1
  %66 = zext i32 %65 to i64
  %67 = icmp ult i64 %64, %66
  br i1 %67, label %85, label %68

68:                                               ; preds = %62
  %69 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %70 = load i64, i64 addrspace(2)* %69, align 8, !tbaa !52
  %71 = lshr i32 %19, 5
  %72 = and i32 %71, 134217726
  %73 = zext i32 %72 to i64
  %74 = icmp uge i64 %70, %73
  %75 = and i64 %70, 1
  %76 = icmp eq i64 %75, 0
  %77 = and i1 %74, %76
  br i1 %77, label %78, label %85

78:                                               ; preds = %68
  %79 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %80 = load i64, i64 addrspace(2)* %79, align 8, !tbaa !53
  %81 = and i64 %80, 1
  %82 = icmp eq i64 %81, 0
  br i1 %82, label %83, label %85

83:                                               ; preds = %78
  %84 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt4ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #13
  br i1 %84, label %90, label %85

85:                                               ; preds = %83, %78, %68, %62, %52, %48, %44, %40, %35, %31, %28, %20, %11
  %86 = icmp eq i32 %10, 0
  br i1 %86, label %87, label %228

87:                                               ; preds = %85
  %88 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %89 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %88, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %228

90:                                               ; preds = %83
  %91 = extractelement <3 x i32> %8, i64 0
  %92 = shl i32 %91, 3
  %93 = shl i32 %9, 2
  %94 = add i32 %92, %93
  %95 = lshr i32 %36, 3
  %96 = icmp uge i32 %91, %95
  %97 = extractelement <3 x i32> %8, i64 1
  %98 = icmp ugt i32 %97, 2
  %99 = or i1 %98, %96
  %100 = extractelement <3 x i32> %8, i64 2
  %101 = icmp ne i32 %100, 0
  %102 = or i1 %101, %99
  %103 = xor i1 %102, true
  %104 = icmp ult i32 %94, %36
  %105 = select i1 %103, i1 %104, i1 false
  br i1 %105, label %106, label %228

106:                                              ; preds = %90
  %107 = icmp eq i32 %97, 2
  br i1 %107, label %108, label %112

108:                                              ; preds = %106
  %109 = zext i32 %19 to i64
  %110 = shl nuw nsw i64 %109, 2
  %111 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %110
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %111, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %94, i32 noundef 0, i64 noundef 4, i32 noundef %10) #12
  br label %228

112:                                              ; preds = %106
  %113 = zext i32 %97 to i64
  %114 = shl nuw nsw i64 %113, 1
  %115 = or i64 %114, 1
  %116 = zext i32 %19 to i64
  %117 = mul nuw nsw i64 %114, %116
  %118 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %117
  %119 = mul i64 %115, %116
  %120 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %119
  %121 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %121) #14
  %122 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %122) #14
  %123 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %123) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %123, i8 0, i64 16, i1 false)
  %124 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %124) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %124, i8 0, i64 16, i1 false)
  %125 = shl i32 %10, 4
  %126 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %127 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %128 = bitcast float* %16 to i8*
  %129 = bitcast float* %17 to i8*
  br label %139

130:                                              ; preds = %151
  %131 = icmp eq i32 %10, 0
  %132 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %133 = zext i32 %36 to i64
  %134 = mul i64 %114, %133
  %135 = zext i32 %94 to i64
  %136 = add i64 %134, %135
  %137 = mul i64 %115, %133
  %138 = add i64 %137, %135
  br label %183

139:                                              ; preds = %151, %112
  %140 = phi i32 [ 0, %112 ], [ %152, %151 ]
  %141 = add i32 %140, %125
  %142 = zext i32 %141 to i64
  %143 = getelementptr inbounds bfloat, bfloat addrspace(1)* %118, i64 %142
  %144 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %143, float* noundef nonnull %126) #13
  %145 = getelementptr inbounds bfloat, bfloat addrspace(1)* %120, i64 %142
  %146 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %145, float* noundef nonnull %127) #13
  %147 = lshr i32 %141, 6
  %148 = lshr exact i64 %142, 1
  %149 = zext i32 %147 to i64
  %150 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %148
  br label %154

151:                                              ; preds = %154
  %152 = add i32 %140, 512
  %153 = icmp ult i32 %152, %19
  br i1 %153, label %139, label %130, !llvm.loop !54

154:                                              ; preds = %154, %139
  %155 = phi i32 [ 0, %139 ], [ %180, %154 ]
  %156 = add nuw nsw i32 %155, %94
  %157 = zext i32 %156 to i64
  %158 = mul i64 %64, %157
  %159 = getelementptr inbounds i8, i8 addrspace(1)* %150, i64 %158
  %160 = mul i64 %70, %157
  %161 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %160
  %162 = bitcast i8 addrspace(1)* %161 to bfloat addrspace(1)*
  %163 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %160
  %164 = bitcast i8 addrspace(1)* %163 to bfloat addrspace(1)*
  %165 = getelementptr inbounds bfloat, bfloat addrspace(1)* %162, i64 %149
  %166 = load bfloat, bfloat addrspace(1)* %165, align 2, !tbaa !56
  %167 = fpext bfloat %166 to float
  %168 = getelementptr inbounds bfloat, bfloat addrspace(1)* %164, i64 %149
  %169 = load bfloat, bfloat addrspace(1)* %168, align 2, !tbaa !56
  %170 = fpext bfloat %169 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %128) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %129) #14
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %159, float* noundef nonnull %126, float* noundef nonnull %127, float noundef %167, float noundef %170, float noundef %144, float noundef %146, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #13
  %171 = load float, float* %16, align 4, !tbaa !58
  %172 = zext i32 %155 to i64
  %173 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %172
  %174 = load float, float* %173, align 4, !tbaa !58
  %175 = fadd float %171, %174
  store float %175, float* %173, align 4, !tbaa !58
  %176 = load float, float* %17, align 4, !tbaa !58
  %177 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %172
  %178 = load float, float* %177, align 4, !tbaa !58
  %179 = fadd float %176, %178
  store float %179, float* %177, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %129) #14
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %128) #14
  %180 = add nuw nsw i32 %155, 1
  %181 = icmp eq i32 %180, 4
  br i1 %181, label %151, label %154, !llvm.loop !60

182:                                              ; preds = %225
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %124) #14
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %123) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %122) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %121) #14
  br label %228

183:                                              ; preds = %225, %130
  %184 = phi i16 [ 0, %130 ], [ %226, %225 ]
  %185 = zext i16 %184 to i64
  %186 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %185
  %187 = load float, float* %186, align 4, !tbaa !58
  %188 = call fast float @air.simd_sum.f32(float %187) #15
  br i1 %131, label %189, label %205

189:                                              ; preds = %183
  %190 = fptrunc float %188 to bfloat
  %191 = bitcast float %188 to i32
  %192 = and i32 %191, 2139095040
  %193 = icmp eq i32 %192, 2139095040
  br i1 %193, label %199, label %194

194:                                              ; preds = %189
  %195 = fpext bfloat %190 to float
  %196 = bitcast float %195 to i32
  %197 = and i32 %196, 2139095040
  %198 = icmp eq i32 %197, 2139095040
  br i1 %198, label %199, label %201

199:                                              ; preds = %194, %189
  %200 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %132, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %201

201:                                              ; preds = %199, %194
  %202 = add i64 %136, %185
  %203 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %202
  store bfloat %190, bfloat addrspace(1)* %203, align 2, !tbaa !56
  %204 = getelementptr inbounds float, float addrspace(1)* %6, i64 %202
  store float %188, float addrspace(1)* %204, align 4, !tbaa !58
  br label %205

205:                                              ; preds = %201, %183
  %206 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %185
  %207 = load float, float* %206, align 4, !tbaa !58
  %208 = call fast float @air.simd_sum.f32(float %207) #15
  br i1 %131, label %209, label %225

209:                                              ; preds = %205
  %210 = fptrunc float %208 to bfloat
  %211 = bitcast float %208 to i32
  %212 = and i32 %211, 2139095040
  %213 = icmp eq i32 %212, 2139095040
  br i1 %213, label %219, label %214

214:                                              ; preds = %209
  %215 = fpext bfloat %210 to float
  %216 = bitcast float %215 to i32
  %217 = and i32 %216, 2139095040
  %218 = icmp eq i32 %217, 2139095040
  br i1 %218, label %219, label %221

219:                                              ; preds = %214, %209
  %220 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %132, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %221

221:                                              ; preds = %219, %214
  %222 = add i64 %138, %185
  %223 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %222
  store bfloat %210, bfloat addrspace(1)* %223, align 2, !tbaa !56
  %224 = getelementptr inbounds float, float addrspace(1)* %6, i64 %222
  store float %208, float addrspace(1)* %224, align 4, !tbaa !58
  br label %225

225:                                              ; preds = %221, %205
  %226 = add nuw nsw i16 %184, 1
  %227 = icmp eq i16 %226, 4
  br i1 %227, label %182, label %183, !llvm.loop !61

228:                                              ; preds = %182, %108, %90, %87, %85
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_candidate_probe_q5_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
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
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca [4 x float], align 4
  %16 = alloca float, align 4
  %17 = alloca float, align 4
  %18 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %19 = load i32, i32 addrspace(2)* %18, align 8, !tbaa !38
  %20 = icmp eq i32 %19, 2560
  br i1 %20, label %21, label %67

21:                                               ; preds = %11
  %22 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %23 = load i32, i32 addrspace(2)* %22, align 4, !tbaa !44
  %24 = icmp eq i32 %23, 10240
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %26 = load i32, i32 addrspace(2)* %25, align 8
  %27 = icmp eq i32 %26, 5
  %28 = select i1 %24, i1 %27, i1 false
  br i1 %28, label %29, label %67

29:                                               ; preds = %21
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %31 = load i32, i32 addrspace(2)* %30, align 4, !tbaa !46
  %32 = icmp eq i32 %31, 1
  br i1 %32, label %33, label %67

33:                                               ; preds = %29
  %34 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %35 = load i32, i32 addrspace(2)* %34, align 8, !tbaa !47
  %36 = icmp eq i32 %35, 1
  br i1 %36, label %37, label %67

37:                                               ; preds = %33
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %39 = load i32, i32 addrspace(2)* %38, align 4, !tbaa !48
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %41, label %67

41:                                               ; preds = %37
  %42 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %43 = load i32, i32 addrspace(2)* %42, align 4, !tbaa !49
  %44 = icmp eq i32 %43, 5
  br i1 %44, label %45, label %67

45:                                               ; preds = %41
  %46 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %47 = load i32, i32 addrspace(2)* %46, align 8, !tbaa !50
  %48 = icmp eq i32 %47, 64
  br i1 %48, label %49, label %67

49:                                               ; preds = %45
  %50 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %51 = load i64, i64 addrspace(2)* %50, align 8, !tbaa !51
  %52 = icmp ult i64 %51, 1600
  br i1 %52, label %67, label %53

53:                                               ; preds = %49
  %54 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %55 = load i64, i64 addrspace(2)* %54, align 8, !tbaa !52
  %56 = icmp ugt i64 %55, 79
  %57 = and i64 %55, 1
  %58 = icmp eq i64 %57, 0
  %59 = and i1 %56, %58
  br i1 %59, label %60, label %67

60:                                               ; preds = %53
  %61 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %62 = load i64, i64 addrspace(2)* %61, align 8, !tbaa !53
  %63 = and i64 %62, 1
  %64 = icmp eq i64 %63, 0
  br i1 %64, label %65, label %67

65:                                               ; preds = %60
  %66 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt5ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #13
  br i1 %66, label %72, label %67

67:                                               ; preds = %65, %60, %53, %49, %45, %41, %37, %33, %29, %21, %11
  %68 = icmp eq i32 %10, 0
  br i1 %68, label %69, label %205

69:                                               ; preds = %67
  %70 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %71 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %70, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %205

72:                                               ; preds = %65
  %73 = extractelement <3 x i32> %8, i64 0
  %74 = shl i32 %73, 3
  %75 = shl i32 %9, 2
  %76 = add i32 %74, %75
  %77 = icmp ugt i32 %73, 1279
  %78 = extractelement <3 x i32> %8, i64 1
  %79 = icmp ugt i32 %78, 2
  %80 = or i1 %79, %77
  %81 = extractelement <3 x i32> %8, i64 2
  %82 = icmp ne i32 %81, 0
  %83 = or i1 %82, %80
  %84 = icmp ugt i32 %76, 10239
  %85 = or i1 %83, %84
  br i1 %85, label %205, label %86

86:                                               ; preds = %72
  %87 = icmp eq i32 %78, 2
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 10240
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %89, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %76, i32 noundef 0, i64 noundef 4, i32 noundef %10) #12
  br label %205

90:                                               ; preds = %86
  %91 = zext i32 %78 to i64
  %92 = shl nuw nsw i64 %91, 1
  %93 = or i64 %92, 1
  %94 = mul nuw nsw i64 %91, 5120
  %95 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %94
  %96 = mul nuw nsw i64 %93, 2560
  %97 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %96
  %98 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %98) #14
  %99 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %99) #14
  %100 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %100) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %100, i8 0, i64 16, i1 false)
  %101 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %101) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %101, i8 0, i64 16, i1 false)
  %102 = shl i32 %10, 4
  %103 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %104 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %105 = bitcast float* %16 to i8*
  %106 = bitcast float* %17 to i8*
  br label %115

107:                                              ; preds = %128
  %108 = icmp eq i32 %10, 0
  %109 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %110 = mul nuw nsw i64 %91, 20480
  %111 = zext i32 %76 to i64
  %112 = add nuw nsw i64 %110, %111
  %113 = mul nuw nsw i64 %93, 10240
  %114 = add nuw nsw i64 %113, %111
  br label %160

115:                                              ; preds = %128, %90
  %116 = phi i32 [ 0, %90 ], [ %129, %128 ]
  %117 = add i32 %116, %102
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds bfloat, bfloat addrspace(1)* %95, i64 %118
  %120 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %119, float* noundef nonnull %103) #13
  %121 = getelementptr inbounds bfloat, bfloat addrspace(1)* %97, i64 %118
  %122 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %121, float* noundef nonnull %104) #13
  %123 = lshr i32 %117, 6
  %124 = mul nuw nsw i64 %118, 5
  %125 = lshr exact i64 %124, 3
  %126 = zext i32 %123 to i64
  %127 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %125
  br label %131

128:                                              ; preds = %131
  %129 = add i32 %116, 512
  %130 = icmp ult i32 %129, 2560
  br i1 %130, label %115, label %107, !llvm.loop !62

131:                                              ; preds = %131, %115
  %132 = phi i32 [ 0, %115 ], [ %157, %131 ]
  %133 = add nuw nsw i32 %132, %76
  %134 = zext i32 %133 to i64
  %135 = mul i64 %51, %134
  %136 = getelementptr inbounds i8, i8 addrspace(1)* %127, i64 %135
  %137 = mul i64 %55, %134
  %138 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %137
  %139 = bitcast i8 addrspace(1)* %138 to bfloat addrspace(1)*
  %140 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %137
  %141 = bitcast i8 addrspace(1)* %140 to bfloat addrspace(1)*
  %142 = getelementptr inbounds bfloat, bfloat addrspace(1)* %139, i64 %126
  %143 = load bfloat, bfloat addrspace(1)* %142, align 2, !tbaa !56
  %144 = fpext bfloat %143 to float
  %145 = getelementptr inbounds bfloat, bfloat addrspace(1)* %141, i64 %126
  %146 = load bfloat, bfloat addrspace(1)* %145, align 2, !tbaa !56
  %147 = fpext bfloat %146 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %105) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %106) #14
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %136, float* noundef nonnull %103, float* noundef nonnull %104, float noundef %144, float noundef %147, float noundef %120, float noundef %122, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #13
  %148 = load float, float* %16, align 4, !tbaa !58
  %149 = zext i32 %132 to i64
  %150 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %149
  %151 = load float, float* %150, align 4, !tbaa !58
  %152 = fadd float %148, %151
  store float %152, float* %150, align 4, !tbaa !58
  %153 = load float, float* %17, align 4, !tbaa !58
  %154 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %149
  %155 = load float, float* %154, align 4, !tbaa !58
  %156 = fadd float %153, %155
  store float %156, float* %154, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %106) #14
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %105) #14
  %157 = add nuw nsw i32 %132, 1
  %158 = icmp eq i32 %157, 4
  br i1 %158, label %128, label %131, !llvm.loop !63

159:                                              ; preds = %202
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %101) #14
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %100) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %99) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %98) #14
  br label %205

160:                                              ; preds = %202, %107
  %161 = phi i16 [ 0, %107 ], [ %203, %202 ]
  %162 = zext i16 %161 to i64
  %163 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %162
  %164 = load float, float* %163, align 4, !tbaa !58
  %165 = call fast float @air.simd_sum.f32(float %164) #15
  br i1 %108, label %166, label %182

166:                                              ; preds = %160
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
  %177 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %109, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %178

178:                                              ; preds = %176, %171
  %179 = add nuw nsw i64 %112, %162
  %180 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %179
  store bfloat %167, bfloat addrspace(1)* %180, align 2, !tbaa !56
  %181 = getelementptr inbounds float, float addrspace(1)* %6, i64 %179
  store float %165, float addrspace(1)* %181, align 4, !tbaa !58
  br label %182

182:                                              ; preds = %178, %160
  %183 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %162
  %184 = load float, float* %183, align 4, !tbaa !58
  %185 = call fast float @air.simd_sum.f32(float %184) #15
  br i1 %108, label %186, label %202

186:                                              ; preds = %182
  %187 = fptrunc float %185 to bfloat
  %188 = bitcast float %185 to i32
  %189 = and i32 %188, 2139095040
  %190 = icmp eq i32 %189, 2139095040
  br i1 %190, label %196, label %191

191:                                              ; preds = %186
  %192 = fpext bfloat %187 to float
  %193 = bitcast float %192 to i32
  %194 = and i32 %193, 2139095040
  %195 = icmp eq i32 %194, 2139095040
  br i1 %195, label %196, label %198

196:                                              ; preds = %191, %186
  %197 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %109, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %198

198:                                              ; preds = %196, %191
  %199 = add nuw nsw i64 %114, %162
  %200 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %199
  store bfloat %187, bfloat addrspace(1)* %200, align 2, !tbaa !56
  %201 = getelementptr inbounds float, float addrspace(1)* %6, i64 %199
  store float %185, float addrspace(1)* %201, align 4, !tbaa !58
  br label %202

202:                                              ; preds = %198, %182
  %203 = add nuw nsw i16 %161, 1
  %204 = icmp eq i16 %203, 4
  br i1 %204, label %159, label %160, !llvm.loop !64

205:                                              ; preds = %159, %88, %72, %69, %67
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_candidate_probe_q5_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
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
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca [4 x float], align 4
  %16 = alloca float, align 4
  %17 = alloca float, align 4
  %18 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %19 = load i32, i32 addrspace(2)* %18, align 8, !tbaa !38
  switch i32 %19, label %84 [
    i32 6144, label %28
    i32 2560, label %20
  ]

20:                                               ; preds = %11
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !44
  %23 = icmp eq i32 %22, 6144
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %25 = load i32, i32 addrspace(2)* %24, align 8
  %26 = icmp eq i32 %25, 5
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %36, label %84

28:                                               ; preds = %11
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !44
  %31 = icmp eq i32 %30, 2560
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %33 = load i32, i32 addrspace(2)* %32, align 8
  %34 = icmp eq i32 %33, 5
  %35 = select i1 %31, i1 %34, i1 false
  br i1 %35, label %36, label %84

36:                                               ; preds = %28, %20
  %37 = phi i32 [ 6144, %20 ], [ 2560, %28 ]
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %39 = load i32, i32 addrspace(2)* %38, align 4, !tbaa !46
  %40 = icmp eq i32 %39, 1
  br i1 %40, label %41, label %84

41:                                               ; preds = %36
  %42 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %43 = load i32, i32 addrspace(2)* %42, align 8, !tbaa !47
  %44 = icmp eq i32 %43, 1
  br i1 %44, label %45, label %84

45:                                               ; preds = %41
  %46 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %47 = load i32, i32 addrspace(2)* %46, align 4, !tbaa !48
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %49, label %84

49:                                               ; preds = %45
  %50 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %51 = load i32, i32 addrspace(2)* %50, align 4, !tbaa !49
  %52 = icmp eq i32 %51, 5
  br i1 %52, label %53, label %84

53:                                               ; preds = %49
  %54 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %55 = load i32, i32 addrspace(2)* %54, align 8, !tbaa !50
  %56 = icmp eq i32 %55, 128
  %57 = and i32 %19, 511
  %58 = icmp eq i32 %57, 0
  %59 = select i1 %56, i1 %58, i1 false
  br i1 %59, label %60, label %84

60:                                               ; preds = %53
  %61 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %62 = load i64, i64 addrspace(2)* %61, align 8, !tbaa !51
  %63 = zext i32 %19 to i64
  %64 = mul nuw nsw i64 %63, 5
  %65 = lshr i64 %64, 3
  %66 = icmp ult i64 %62, %65
  br i1 %66, label %84, label %67

67:                                               ; preds = %60
  %68 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %69 = load i64, i64 addrspace(2)* %68, align 8, !tbaa !52
  %70 = lshr i32 %19, 6
  %71 = and i32 %70, 67108862
  %72 = zext i32 %71 to i64
  %73 = icmp uge i64 %69, %72
  %74 = and i64 %69, 1
  %75 = icmp eq i64 %74, 0
  %76 = and i1 %73, %75
  br i1 %76, label %77, label %84

77:                                               ; preds = %67
  %78 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %79 = load i64, i64 addrspace(2)* %78, align 8, !tbaa !53
  %80 = and i64 %79, 1
  %81 = icmp eq i64 %80, 0
  br i1 %81, label %82, label %84

82:                                               ; preds = %77
  %83 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt5ELt128EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #13
  br i1 %83, label %89, label %84

84:                                               ; preds = %82, %77, %67, %60, %53, %49, %45, %41, %36, %28, %20, %11
  %85 = icmp eq i32 %10, 0
  br i1 %85, label %86, label %225

86:                                               ; preds = %84
  %87 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %88 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %87, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %225

89:                                               ; preds = %82
  %90 = extractelement <3 x i32> %8, i64 0
  %91 = shl i32 %90, 3
  %92 = shl i32 %9, 2
  %93 = add i32 %91, %92
  %94 = lshr exact i32 %37, 3
  %95 = icmp uge i32 %90, %94
  %96 = extractelement <3 x i32> %8, i64 1
  %97 = icmp ugt i32 %96, 2
  %98 = or i1 %97, %95
  %99 = extractelement <3 x i32> %8, i64 2
  %100 = icmp ne i32 %99, 0
  %101 = or i1 %100, %98
  %102 = icmp uge i32 %93, %37
  %103 = or i1 %101, %102
  br i1 %103, label %225, label %104

104:                                              ; preds = %89
  %105 = icmp eq i32 %96, 2
  br i1 %105, label %106, label %109

106:                                              ; preds = %104
  %107 = shl nuw nsw i64 %63, 2
  %108 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %107
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %108, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %93, i32 noundef 0, i64 noundef 4, i32 noundef %10) #12
  br label %225

109:                                              ; preds = %104
  %110 = zext i32 %96 to i64
  %111 = shl nuw nsw i64 %110, 1
  %112 = or i64 %111, 1
  %113 = mul nuw nsw i64 %111, %63
  %114 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %113
  %115 = mul i64 %112, %63
  %116 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %115
  %117 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %117) #14
  %118 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %118) #14
  %119 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %119) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %119, i8 0, i64 16, i1 false)
  %120 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %120) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %120, i8 0, i64 16, i1 false)
  %121 = shl i32 %10, 4
  %122 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %123 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %124 = bitcast float* %16 to i8*
  %125 = bitcast float* %17 to i8*
  br label %135

126:                                              ; preds = %148
  %127 = icmp eq i32 %10, 0
  %128 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %129 = zext i32 %37 to i64
  %130 = mul nuw nsw i64 %111, %129
  %131 = zext i32 %93 to i64
  %132 = add nuw nsw i64 %130, %131
  %133 = mul nuw nsw i64 %112, %129
  %134 = add nuw nsw i64 %133, %131
  br label %180

135:                                              ; preds = %148, %109
  %136 = phi i32 [ 0, %109 ], [ %149, %148 ]
  %137 = add i32 %136, %121
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds bfloat, bfloat addrspace(1)* %114, i64 %138
  %140 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %139, float* noundef nonnull %122) #13
  %141 = getelementptr inbounds bfloat, bfloat addrspace(1)* %116, i64 %138
  %142 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %141, float* noundef nonnull %123) #13
  %143 = lshr i32 %137, 7
  %144 = mul nuw nsw i64 %138, 5
  %145 = lshr exact i64 %144, 3
  %146 = zext i32 %143 to i64
  %147 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %145
  br label %151

148:                                              ; preds = %151
  %149 = add i32 %136, 512
  %150 = icmp ult i32 %149, %19
  br i1 %150, label %135, label %126, !llvm.loop !65

151:                                              ; preds = %151, %135
  %152 = phi i32 [ 0, %135 ], [ %177, %151 ]
  %153 = add nuw nsw i32 %152, %93
  %154 = zext i32 %153 to i64
  %155 = mul i64 %62, %154
  %156 = getelementptr inbounds i8, i8 addrspace(1)* %147, i64 %155
  %157 = mul i64 %69, %154
  %158 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %157
  %159 = bitcast i8 addrspace(1)* %158 to bfloat addrspace(1)*
  %160 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %157
  %161 = bitcast i8 addrspace(1)* %160 to bfloat addrspace(1)*
  %162 = getelementptr inbounds bfloat, bfloat addrspace(1)* %159, i64 %146
  %163 = load bfloat, bfloat addrspace(1)* %162, align 2, !tbaa !56
  %164 = fpext bfloat %163 to float
  %165 = getelementptr inbounds bfloat, bfloat addrspace(1)* %161, i64 %146
  %166 = load bfloat, bfloat addrspace(1)* %165, align 2, !tbaa !56
  %167 = fpext bfloat %166 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %124) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %125) #14
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %156, float* noundef nonnull %122, float* noundef nonnull %123, float noundef %164, float noundef %167, float noundef %140, float noundef %142, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #13
  %168 = load float, float* %16, align 4, !tbaa !58
  %169 = zext i32 %152 to i64
  %170 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %169
  %171 = load float, float* %170, align 4, !tbaa !58
  %172 = fadd float %168, %171
  store float %172, float* %170, align 4, !tbaa !58
  %173 = load float, float* %17, align 4, !tbaa !58
  %174 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %169
  %175 = load float, float* %174, align 4, !tbaa !58
  %176 = fadd float %173, %175
  store float %176, float* %174, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %125) #14
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %124) #14
  %177 = add nuw nsw i32 %152, 1
  %178 = icmp eq i32 %177, 4
  br i1 %178, label %148, label %151, !llvm.loop !66

179:                                              ; preds = %222
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %120) #14
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %119) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %118) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %117) #14
  br label %225

180:                                              ; preds = %222, %126
  %181 = phi i16 [ 0, %126 ], [ %223, %222 ]
  %182 = zext i16 %181 to i64
  %183 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %182
  %184 = load float, float* %183, align 4, !tbaa !58
  %185 = call fast float @air.simd_sum.f32(float %184) #15
  br i1 %127, label %186, label %202

186:                                              ; preds = %180
  %187 = fptrunc float %185 to bfloat
  %188 = bitcast float %185 to i32
  %189 = and i32 %188, 2139095040
  %190 = icmp eq i32 %189, 2139095040
  br i1 %190, label %196, label %191

191:                                              ; preds = %186
  %192 = fpext bfloat %187 to float
  %193 = bitcast float %192 to i32
  %194 = and i32 %193, 2139095040
  %195 = icmp eq i32 %194, 2139095040
  br i1 %195, label %196, label %198

196:                                              ; preds = %191, %186
  %197 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %128, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %198

198:                                              ; preds = %196, %191
  %199 = add nuw nsw i64 %132, %182
  %200 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %199
  store bfloat %187, bfloat addrspace(1)* %200, align 2, !tbaa !56
  %201 = getelementptr inbounds float, float addrspace(1)* %6, i64 %199
  store float %185, float addrspace(1)* %201, align 4, !tbaa !58
  br label %202

202:                                              ; preds = %198, %180
  %203 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %182
  %204 = load float, float* %203, align 4, !tbaa !58
  %205 = call fast float @air.simd_sum.f32(float %204) #15
  br i1 %127, label %206, label %222

206:                                              ; preds = %202
  %207 = fptrunc float %205 to bfloat
  %208 = bitcast float %205 to i32
  %209 = and i32 %208, 2139095040
  %210 = icmp eq i32 %209, 2139095040
  br i1 %210, label %216, label %211

211:                                              ; preds = %206
  %212 = fpext bfloat %207 to float
  %213 = bitcast float %212 to i32
  %214 = and i32 %213, 2139095040
  %215 = icmp eq i32 %214, 2139095040
  br i1 %215, label %216, label %218

216:                                              ; preds = %211, %206
  %217 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %128, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %218

218:                                              ; preds = %216, %211
  %219 = add nuw nsw i64 %134, %182
  %220 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %219
  store bfloat %207, bfloat addrspace(1)* %220, align 2, !tbaa !56
  %221 = getelementptr inbounds float, float addrspace(1)* %6, i64 %219
  store float %205, float addrspace(1)* %221, align 4, !tbaa !58
  br label %222

222:                                              ; preds = %218, %202
  %223 = add nuw nsw i16 %181, 1
  %224 = icmp eq i16 %223, 4
  br i1 %224, label %179, label %180, !llvm.loop !67

225:                                              ; preds = %179, %106, %89, %86, %84
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_candidate_probe_q6_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
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
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #12
  br label %29

29:                                               ; preds = %28, %25, %23
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %9, i32 noundef %10) local_unnamed_addr #1 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [8 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca [4 x float], align 4
  %16 = alloca float, align 4
  %17 = alloca float, align 4
  %18 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %19 = load i32, i32 addrspace(2)* %18, align 8, !tbaa !38
  %20 = icmp eq i32 %19, 2560
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4
  %23 = icmp eq i32 %22, 6144
  %24 = select i1 %20, i1 %23, i1 false
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %26 = load i32, i32 addrspace(2)* %25, align 8
  %27 = icmp eq i32 %26, 5
  %28 = select i1 %24, i1 %27, i1 false
  br i1 %28, label %29, label %67

29:                                               ; preds = %11
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 1
  %31 = load i32, i32 addrspace(2)* %30, align 4, !tbaa !46
  %32 = icmp eq i32 %31, 1
  br i1 %32, label %33, label %67

33:                                               ; preds = %29
  %34 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 4
  %35 = load i32, i32 addrspace(2)* %34, align 8, !tbaa !47
  %36 = icmp eq i32 %35, 1
  br i1 %36, label %37, label %67

37:                                               ; preds = %33
  %38 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 7
  %39 = load i32, i32 addrspace(2)* %38, align 4, !tbaa !48
  %40 = icmp eq i32 %39, 0
  br i1 %40, label %41, label %67

41:                                               ; preds = %37
  %42 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 5
  %43 = load i32, i32 addrspace(2)* %42, align 4, !tbaa !49
  %44 = icmp eq i32 %43, 6
  br i1 %44, label %45, label %67

45:                                               ; preds = %41
  %46 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 6
  %47 = load i32, i32 addrspace(2)* %46, align 8, !tbaa !50
  %48 = icmp eq i32 %47, 64
  br i1 %48, label %49, label %67

49:                                               ; preds = %45
  %50 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 8
  %51 = load i64, i64 addrspace(2)* %50, align 8, !tbaa !51
  %52 = icmp ult i64 %51, 1920
  br i1 %52, label %67, label %53

53:                                               ; preds = %49
  %54 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 10
  %55 = load i64, i64 addrspace(2)* %54, align 8, !tbaa !52
  %56 = icmp ugt i64 %55, 79
  %57 = and i64 %55, 1
  %58 = icmp eq i64 %57, 0
  %59 = and i1 %56, %58
  br i1 %59, label %60, label %67

60:                                               ; preds = %53
  %61 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %62 = load i64, i64 addrspace(2)* %61, align 8, !tbaa !53
  %63 = and i64 %62, 1
  %64 = icmp eq i64 %63, 0
  br i1 %64, label %65, label %67

65:                                               ; preds = %60
  %66 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt6ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #13
  br i1 %66, label %72, label %67

67:                                               ; preds = %65, %60, %53, %49, %45, %41, %37, %33, %29, %11
  %68 = icmp eq i32 %10, 0
  br i1 %68, label %69, label %205

69:                                               ; preds = %67
  %70 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %71 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %70, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %205

72:                                               ; preds = %65
  %73 = extractelement <3 x i32> %8, i64 0
  %74 = shl i32 %73, 3
  %75 = shl i32 %9, 2
  %76 = add i32 %74, %75
  %77 = icmp ugt i32 %73, 767
  %78 = extractelement <3 x i32> %8, i64 1
  %79 = icmp ugt i32 %78, 2
  %80 = or i1 %79, %77
  %81 = extractelement <3 x i32> %8, i64 2
  %82 = icmp ne i32 %81, 0
  %83 = or i1 %82, %80
  %84 = icmp ugt i32 %76, 6143
  %85 = or i1 %83, %84
  br i1 %85, label %205, label %86

86:                                               ; preds = %72
  %87 = icmp eq i32 %78, 2
  br i1 %87, label %88, label %90

88:                                               ; preds = %86
  %89 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 10240
  tail call void @_ZN22r5_raw_odd_control_tap12project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %89, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %76, i32 noundef 0, i64 noundef 4, i32 noundef %10) #12
  br label %205

90:                                               ; preds = %86
  %91 = zext i32 %78 to i64
  %92 = shl nuw nsw i64 %91, 1
  %93 = or i64 %92, 1
  %94 = mul nuw nsw i64 %91, 5120
  %95 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %94
  %96 = mul nuw nsw i64 %93, 2560
  %97 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %96
  %98 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %98) #14
  %99 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %99) #14
  %100 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %100) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %100, i8 0, i64 16, i1 false)
  %101 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %101) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %101, i8 0, i64 16, i1 false)
  %102 = shl i32 %10, 3
  %103 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %104 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %105 = bitcast float* %16 to i8*
  %106 = bitcast float* %17 to i8*
  br label %115

107:                                              ; preds = %128
  %108 = icmp eq i32 %10, 0
  %109 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %110 = mul nuw nsw i64 %91, 12288
  %111 = zext i32 %76 to i64
  %112 = add nuw nsw i64 %110, %111
  %113 = mul nuw nsw i64 %93, 6144
  %114 = add nuw nsw i64 %113, %111
  br label %160

115:                                              ; preds = %128, %90
  %116 = phi i32 [ 0, %90 ], [ %129, %128 ]
  %117 = add i32 %116, %102
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds bfloat, bfloat addrspace(1)* %95, i64 %118
  %120 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %119, float* noundef nonnull %103) #13
  %121 = getelementptr inbounds bfloat, bfloat addrspace(1)* %97, i64 %118
  %122 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %121, float* noundef nonnull %104) #13
  %123 = lshr i32 %117, 6
  %124 = mul nuw nsw i64 %118, 6
  %125 = lshr exact i64 %124, 3
  %126 = zext i32 %123 to i64
  %127 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %125
  br label %131

128:                                              ; preds = %131
  %129 = add i32 %116, 256
  %130 = icmp ult i32 %129, 2560
  br i1 %130, label %115, label %107, !llvm.loop !68

131:                                              ; preds = %131, %115
  %132 = phi i32 [ 0, %115 ], [ %157, %131 ]
  %133 = add nuw nsw i32 %132, %76
  %134 = zext i32 %133 to i64
  %135 = mul i64 %51, %134
  %136 = getelementptr inbounds i8, i8 addrspace(1)* %127, i64 %135
  %137 = mul i64 %55, %134
  %138 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %137
  %139 = bitcast i8 addrspace(1)* %138 to bfloat addrspace(1)*
  %140 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %137
  %141 = bitcast i8 addrspace(1)* %140 to bfloat addrspace(1)*
  %142 = getelementptr inbounds bfloat, bfloat addrspace(1)* %139, i64 %126
  %143 = load bfloat, bfloat addrspace(1)* %142, align 2, !tbaa !56
  %144 = fpext bfloat %143 to float
  %145 = getelementptr inbounds bfloat, bfloat addrspace(1)* %141, i64 %126
  %146 = load bfloat, bfloat addrspace(1)* %145, align 2, !tbaa !56
  %147 = fpext bfloat %146 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %105) #14
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %106) #14
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %136, float* noundef nonnull %103, float* noundef nonnull %104, float noundef %144, float noundef %147, float noundef %120, float noundef %122, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #13
  %148 = load float, float* %16, align 4, !tbaa !58
  %149 = zext i32 %132 to i64
  %150 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %149
  %151 = load float, float* %150, align 4, !tbaa !58
  %152 = fadd float %148, %151
  store float %152, float* %150, align 4, !tbaa !58
  %153 = load float, float* %17, align 4, !tbaa !58
  %154 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %149
  %155 = load float, float* %154, align 4, !tbaa !58
  %156 = fadd float %153, %155
  store float %156, float* %154, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %106) #14
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %105) #14
  %157 = add nuw nsw i32 %132, 1
  %158 = icmp eq i32 %157, 4
  br i1 %158, label %128, label %131, !llvm.loop !69

159:                                              ; preds = %202
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %101) #14
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %100) #14
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %99) #14
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %98) #14
  br label %205

160:                                              ; preds = %202, %107
  %161 = phi i16 [ 0, %107 ], [ %203, %202 ]
  %162 = zext i16 %161 to i64
  %163 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %162
  %164 = load float, float* %163, align 4, !tbaa !58
  %165 = call fast float @air.simd_sum.f32(float %164) #15
  br i1 %108, label %166, label %182

166:                                              ; preds = %160
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
  %177 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %109, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %178

178:                                              ; preds = %176, %171
  %179 = add nuw nsw i64 %112, %162
  %180 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %179
  store bfloat %167, bfloat addrspace(1)* %180, align 2, !tbaa !56
  %181 = getelementptr inbounds float, float addrspace(1)* %6, i64 %179
  store float %165, float addrspace(1)* %181, align 4, !tbaa !58
  br label %182

182:                                              ; preds = %178, %160
  %183 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %162
  %184 = load float, float* %183, align 4, !tbaa !58
  %185 = call fast float @air.simd_sum.f32(float %184) #15
  br i1 %108, label %186, label %202

186:                                              ; preds = %182
  %187 = fptrunc float %185 to bfloat
  %188 = bitcast float %185 to i32
  %189 = and i32 %188, 2139095040
  %190 = icmp eq i32 %189, 2139095040
  br i1 %190, label %196, label %191

191:                                              ; preds = %186
  %192 = fpext bfloat %187 to float
  %193 = bitcast float %192 to i32
  %194 = and i32 %193, 2139095040
  %195 = icmp eq i32 %194, 2139095040
  br i1 %195, label %196, label %198

196:                                              ; preds = %191, %186
  %197 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %109, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %198

198:                                              ; preds = %196, %191
  %199 = add nuw nsw i64 %114, %162
  %200 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %199
  store bfloat %187, bfloat addrspace(1)* %200, align 2, !tbaa !56
  %201 = getelementptr inbounds float, float addrspace(1)* %6, i64 %199
  store float %185, float addrspace(1)* %201, align 4, !tbaa !58
  br label %202

202:                                              ; preds = %198, %182
  %203 = add nuw nsw i16 %161, 1
  %204 = icmp eq i16 %203, 4
  br i1 %204, label %159, label %160, !llvm.loop !70

205:                                              ; preds = %159, %88, %72, %69, %67
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i1 @air.any.v3i1(<3 x i1>) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture, i32, i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt4ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !38
  switch i32 %3, label %25 [
    i32 2560, label %4
    i32 6144, label %28
  ]

4:                                                ; preds = %1
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !44
  switch i32 %6, label %23 [
    i32 10240, label %7
    i32 12288, label %15
  ]

7:                                                ; preds = %4
  %8 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %9 = load i64, i64 addrspace(2)* %8, align 8, !tbaa !51
  %10 = icmp ult i64 %9, 1801615789990190
  %11 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %12 = load i64, i64 addrspace(2)* %11, align 8
  %13 = icmp ult i64 %12, 1801615789990190
  %14 = select i1 %10, i1 %13, i1 false
  br label %62

15:                                               ; preds = %4
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %17 = load i64, i64 addrspace(2)* %16, align 8, !tbaa !51
  %18 = icmp ult i64 %17, 1501322053691671
  %19 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %20 = load i64, i64 addrspace(2)* %19, align 8
  %21 = icmp ult i64 %20, 1501322053691671
  %22 = select i1 %18, i1 %21, i1 false
  br label %62

23:                                               ; preds = %4
  %24 = icmp eq i32 %3, 6144
  br i1 %24, label %28, label %25

25:                                               ; preds = %23, %1
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %27 = load i32, i32 addrspace(2)* %26, align 4, !tbaa !44
  br label %40

28:                                               ; preds = %23, %1
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !44
  %31 = icmp eq i32 %30, 2560
  br i1 %31, label %32, label %40

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %34 = load i64, i64 addrspace(2)* %33, align 8, !tbaa !51
  %35 = icmp ult i64 %34, 7208575253501192
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %37 = load i64, i64 addrspace(2)* %36, align 8
  %38 = icmp ult i64 %37, 7208575253501193
  %39 = select i1 %35, i1 %38, i1 false
  br label %62

40:                                               ; preds = %28, %25
  %41 = phi i32 [ %27, %25 ], [ %30, %28 ]
  %42 = icmp ult i32 %41, 2
  br i1 %42, label %62, label %43

43:                                               ; preds = %40
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %45 = load i64, i64 addrspace(2)* %44, align 8, !tbaa !51
  %46 = lshr i32 %3, 1
  %47 = zext i32 %46 to i64
  %48 = xor i64 %47, -1
  %49 = add i32 %41, -1
  %50 = zext i32 %49 to i64
  %51 = udiv i64 %48, %50
  %52 = icmp ugt i64 %45, %51
  br i1 %52, label %62, label %53

53:                                               ; preds = %43
  %54 = lshr i32 %3, 5
  %55 = and i32 %54, 134217726
  %56 = zext i32 %55 to i64
  %57 = xor i64 %56, -1
  %58 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %59 = load i64, i64 addrspace(2)* %58, align 8, !tbaa !52
  %60 = udiv i64 %57, %50
  %61 = icmp ule i64 %59, %60
  br label %62

62:                                               ; preds = %53, %43, %40, %32, %15, %7
  %63 = phi i1 [ %14, %7 ], [ %22, %15 ], [ %39, %32 ], [ false, %40 ], [ false, %43 ], [ %61, %53 ]
  ret i1 %63
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #6 {
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #14
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !38
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %24, label %20

20:                                               ; preds = %12
  %21 = shl i32 %11, 4
  %22 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  br label %33

24:                                               ; preds = %73, %12
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !44
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
  %43 = load bfloat, bfloat addrspace(1)* %42, align 2, !tbaa !56
  %44 = fpext bfloat %43 to float
  %45 = or i32 %39, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !56
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %39, 2
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !56
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = or i32 %39, 3
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %58
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !56
  %61 = fpext bfloat %60 to float
  %62 = fadd float %56, %61
  %63 = fadd float %40, %62
  %64 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %41
  store float %44, float* %64, align 4, !tbaa !58
  %65 = fmul float %49, 6.250000e-02
  %66 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %46
  store float %65, float* %66, align 4, !tbaa !58
  %67 = fmul float %55, 3.906250e-03
  %68 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %52
  store float %67, float* %68, align 4, !tbaa !58
  %69 = fmul float %61, 0x3F30000000000000
  %70 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %58
  store float %69, float* %70, align 4, !tbaa !58
  %71 = add nuw nsw i32 %39, 4
  %72 = icmp ult i32 %39, 12
  br i1 %72, label %38, label %73, !llvm.loop !71

73:                                               ; preds = %38
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %35, i32 noundef %9, float* noundef nonnull %22, float noundef %63, i32 noundef 16, i1 noundef zeroext false, float* noundef nonnull %23) #13
  %74 = add i32 %34, 512
  %75 = icmp ult i32 %74, %18
  br i1 %75, label %33, label %24, !llvm.loop !72

76:                                               ; preds = %102
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #14
  ret void

77:                                               ; preds = %102, %24
  %78 = phi i32 [ 0, %24 ], [ %103, %102 ]
  %79 = add i32 %78, %8
  %80 = icmp ult i32 %79, %26
  br i1 %80, label %81, label %102

81:                                               ; preds = %77
  %82 = zext i32 %78 to i64
  %83 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %82
  %84 = load float, float* %83, align 4, !tbaa !58
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
  store bfloat %87, bfloat addrspace(1)* %100, align 2, !tbaa !56
  %101 = getelementptr inbounds float, float addrspace(1)* %6, i64 %99
  store float %85, float addrspace(1)* %101, align 4, !tbaa !58
  br label %102

102:                                              ; preds = %98, %81, %77
  %103 = add nuw nsw i32 %78, 1
  %104 = icmp eq i32 %103, 4
  br i1 %104, label %76, label %77, !llvm.loop !73
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #7

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #5 {
  br label %4

3:                                                ; preds = %4
  ret float %29

4:                                                ; preds = %4, %2
  %5 = phi i32 [ 0, %2 ], [ %37, %4 ]
  %6 = phi float [ 0.000000e+00, %2 ], [ %29, %4 ]
  %7 = zext i32 %5 to i64
  %8 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %7
  %9 = load bfloat, bfloat addrspace(1)* %8, align 2, !tbaa !56
  %10 = fpext bfloat %9 to float
  %11 = or i32 %5, 1
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %12
  %14 = load bfloat, bfloat addrspace(1)* %13, align 2, !tbaa !56
  %15 = fpext bfloat %14 to float
  %16 = fadd float %10, %15
  %17 = or i32 %5, 2
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %18
  %20 = load bfloat, bfloat addrspace(1)* %19, align 2, !tbaa !56
  %21 = fpext bfloat %20 to float
  %22 = fadd float %16, %21
  %23 = or i32 %5, 3
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %24
  %26 = load bfloat, bfloat addrspace(1)* %25, align 2, !tbaa !56
  %27 = fpext bfloat %26 to float
  %28 = fadd float %22, %27
  %29 = fadd float %6, %28
  %30 = getelementptr inbounds float, float* %1, i64 %7
  store float %10, float* %30, align 4, !tbaa !58
  %31 = fmul float %15, 6.250000e-02
  %32 = getelementptr inbounds float, float* %1, i64 %12
  store float %31, float* %32, align 4, !tbaa !58
  %33 = fmul float %21, 3.906250e-03
  %34 = getelementptr inbounds float, float* %1, i64 %18
  store float %33, float* %34, align 4, !tbaa !58
  %35 = fmul float %27, 0x3F30000000000000
  %36 = getelementptr inbounds float, float* %1, i64 %24
  store float %35, float* %36, align 4, !tbaa !58
  %37 = add nuw nsw i32 %5, 4
  %38 = icmp ult i32 %5, 12
  br i1 %38, label %4, label %3, !llvm.loop !71
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #5 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %59, float %11)
  store float %12, float* %7, align 4, !tbaa !58
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %72, float %13)
  store float %14, float* %8, align 4, !tbaa !58
  ret void

15:                                               ; preds = %15, %9
  %16 = phi float [ 0.000000e+00, %9 ], [ %59, %15 ]
  %17 = phi float [ 0.000000e+00, %9 ], [ %72, %15 ]
  %18 = phi i32 [ 0, %9 ], [ %73, %15 ]
  %19 = shl nuw nsw i32 %18, 1
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %20
  %22 = load i8, i8 addrspace(1)* %21, align 1, !tbaa !74
  %23 = or i32 %19, 1
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %24
  %26 = load i8, i8 addrspace(1)* %25, align 1, !tbaa !74
  %27 = zext i8 %26 to i32
  %28 = shl nuw nsw i32 %27, 8
  %29 = and i8 %22, 15
  %30 = and i8 %22, -16
  %31 = and i32 %28, 3840
  %32 = and i32 %28, 61440
  %33 = shl nuw nsw i32 %18, 2
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds float, float* %1, i64 %34
  %36 = load float, float* %35, align 4, !tbaa !58
  %37 = zext i8 %29 to i32
  %38 = tail call float @air.convert.f.f32.s.i32(i32 %37) #10
  %39 = or i32 %33, 1
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds float, float* %1, i64 %40
  %42 = load float, float* %41, align 4, !tbaa !58
  %43 = zext i8 %30 to i32
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %43) #10
  %45 = fmul float %42, %44
  %46 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %45)
  %47 = or i32 %33, 2
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds float, float* %1, i64 %48
  %50 = load float, float* %49, align 4, !tbaa !58
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %31) #10
  %52 = tail call float @llvm.fmuladd.f32(float %50, float %51, float %46)
  %53 = or i32 %33, 3
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds float, float* %1, i64 %54
  %56 = load float, float* %55, align 4, !tbaa !58
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %32) #10
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %52)
  %59 = fadd float %16, %58
  %60 = getelementptr inbounds float, float* %2, i64 %34
  %61 = load float, float* %60, align 4, !tbaa !58
  %62 = getelementptr inbounds float, float* %2, i64 %40
  %63 = load float, float* %62, align 4, !tbaa !58
  %64 = fmul float %44, %63
  %65 = tail call float @llvm.fmuladd.f32(float %61, float %38, float %64)
  %66 = getelementptr inbounds float, float* %2, i64 %48
  %67 = load float, float* %66, align 4, !tbaa !58
  %68 = tail call float @llvm.fmuladd.f32(float %67, float %51, float %65)
  %69 = getelementptr inbounds float, float* %2, i64 %54
  %70 = load float, float* %69, align 4, !tbaa !58
  %71 = tail call float @llvm.fmuladd.f32(float %70, float %57, float %68)
  %72 = fadd float %17, %71
  %73 = add nuw nsw i32 %18, 1
  %74 = icmp eq i32 %73, 4
  br i1 %74, label %10, label %15, !llvm.loop !75
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #5 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !53
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !76
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !44
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
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !74
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !74
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !58
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !58
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #10
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #14
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !58
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #10
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #14
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !58
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #10
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #14
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !77

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #14
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !58
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !58
  br label %161

109:                                              ; preds = %109, %39
  %110 = phi float [ %151, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i32 [ %152, %109 ], [ 0, %39 ]
  %112 = shl nuw nsw i32 %111, 1
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %113
  %115 = load i8, i8 addrspace(1)* %114, align 1, !tbaa !74
  %116 = zext i8 %115 to i32
  %117 = or i32 %112, 1
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %118
  %120 = load i8, i8 addrspace(1)* %119, align 1, !tbaa !74
  %121 = zext i8 %120 to i32
  %122 = shl nuw nsw i32 %121, 8
  %123 = shl nuw nsw i32 %111, 2
  %124 = zext i32 %123 to i64
  %125 = getelementptr inbounds float, float* %7, i64 %124
  %126 = load float, float* %125, align 4, !tbaa !58
  %127 = and i32 %116, 15
  %128 = tail call float @air.convert.f.f32.s.i32(i32 %127) #10
  %129 = or i32 %123, 1
  %130 = zext i32 %129 to i64
  %131 = getelementptr inbounds float, float* %7, i64 %130
  %132 = load float, float* %131, align 4, !tbaa !58
  %133 = and i32 %116, 240
  %134 = tail call float @air.convert.f.f32.s.i32(i32 %133) #10
  %135 = fmul float %132, %134
  %136 = tail call float @llvm.fmuladd.f32(float %126, float %128, float %135) #14
  %137 = or i32 %123, 2
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds float, float* %7, i64 %138
  %140 = load float, float* %139, align 4, !tbaa !58
  %141 = and i32 %122, 3840
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #10
  %143 = tail call float @llvm.fmuladd.f32(float %140, float %142, float %136) #14
  %144 = or i32 %123, 3
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds float, float* %7, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !58
  %148 = and i32 %122, 61440
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #10
  %150 = tail call float @llvm.fmuladd.f32(float %147, float %149, float %143) #14
  %151 = fadd float %110, %150
  %152 = add nuw nsw i32 %111, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %154, label %109, !llvm.loop !78

154:                                              ; preds = %109
  %155 = fmul float %54, %8
  %156 = tail call float @llvm.fmuladd.f32(float %51, float %151, float %155) #14
  %157 = zext i32 %36 to i64
  %158 = getelementptr inbounds float, float* %11, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !58
  %160 = fadd float %159, %156
  store float %160, float* %158, align 4, !tbaa !58
  br label %161

161:                                              ; preds = %154, %101, %35
  %162 = add nuw nsw i32 %36, 1
  %163 = icmp eq i32 %162, 4
  br i1 %163, label %34, label %35, !llvm.loop !79
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #8

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #9

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt5ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !38
  %4 = icmp eq i32 %3, 2560
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !44
  %7 = icmp eq i32 %6, 10240
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %9, label %17

9:                                                ; preds = %1
  %10 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %11 = load i64, i64 addrspace(2)* %10, align 8, !tbaa !51
  %12 = icmp ult i64 %11, 1801615789990190
  %13 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %14 = load i64, i64 addrspace(2)* %13, align 8
  %15 = icmp ult i64 %14, 1801615789990190
  %16 = select i1 %12, i1 %15, i1 false
  br label %39

17:                                               ; preds = %1
  %18 = icmp ult i32 %6, 2
  br i1 %18, label %39, label %19

19:                                               ; preds = %17
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %21 = load i64, i64 addrspace(2)* %20, align 8, !tbaa !51
  %22 = zext i32 %3 to i64
  %23 = mul nuw nsw i64 %22, 5
  %24 = lshr i64 %23, 3
  %25 = xor i64 %24, -1
  %26 = add i32 %6, -1
  %27 = zext i32 %26 to i64
  %28 = udiv i64 %25, %27
  %29 = icmp ugt i64 %21, %28
  br i1 %29, label %39, label %30

30:                                               ; preds = %19
  %31 = lshr i32 %3, 5
  %32 = and i32 %31, 134217726
  %33 = zext i32 %32 to i64
  %34 = xor i64 %33, -1
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %36 = load i64, i64 addrspace(2)* %35, align 8, !tbaa !52
  %37 = udiv i64 %34, %27
  %38 = icmp ule i64 %36, %37
  br label %39

39:                                               ; preds = %30, %19, %17, %9
  %40 = phi i1 [ %16, %9 ], [ false, %17 ], [ false, %19 ], [ %38, %30 ]
  ret i1 %40
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #6 {
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #14
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !38
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %23

20:                                               ; preds = %12
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !44
  br label %40

23:                                               ; preds = %12
  %24 = shl i32 %11, 4
  %25 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %26 = zext i32 %9 to i64
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %28 = load i64, i64 addrspace(2)* %27, align 8, !tbaa !53
  %29 = mul i64 %28, %26
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %31 = load i64, i64 addrspace(2)* %30, align 8, !tbaa !76
  %32 = mul i64 %31, %26
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !44
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
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !56
  %60 = fpext bfloat %59 to float
  %61 = or i32 %55, 1
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %62
  %64 = load bfloat, bfloat addrspace(1)* %63, align 2, !tbaa !56
  %65 = fpext bfloat %64 to float
  %66 = fadd float %60, %65
  %67 = or i32 %55, 2
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %68
  %70 = load bfloat, bfloat addrspace(1)* %69, align 2, !tbaa !56
  %71 = fpext bfloat %70 to float
  %72 = fadd float %66, %71
  %73 = or i32 %55, 3
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %74
  %76 = load bfloat, bfloat addrspace(1)* %75, align 2, !tbaa !56
  %77 = fpext bfloat %76 to float
  %78 = fadd float %72, %77
  %79 = or i32 %55, 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %80
  %82 = load bfloat, bfloat addrspace(1)* %81, align 2, !tbaa !56
  %83 = fpext bfloat %82 to float
  %84 = fadd float %78, %83
  %85 = or i32 %55, 5
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %86
  %88 = load bfloat, bfloat addrspace(1)* %87, align 2, !tbaa !56
  %89 = fpext bfloat %88 to float
  %90 = fadd float %84, %89
  %91 = or i32 %55, 6
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %92
  %94 = load bfloat, bfloat addrspace(1)* %93, align 2, !tbaa !56
  %95 = fpext bfloat %94 to float
  %96 = fadd float %90, %95
  %97 = or i32 %55, 7
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %98
  %100 = load bfloat, bfloat addrspace(1)* %99, align 2, !tbaa !56
  %101 = fpext bfloat %100 to float
  %102 = fadd float %96, %101
  %103 = fadd float %56, %102
  %104 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %57
  store float %60, float* %104, align 4, !tbaa !58
  %105 = fmul float %65, 3.125000e-02
  %106 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %62
  store float %105, float* %106, align 4, !tbaa !58
  %107 = fmul float %71, 2.500000e-01
  %108 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %68
  store float %107, float* %108, align 4, !tbaa !58
  %109 = fmul float %77, 7.812500e-03
  %110 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %74
  store float %109, float* %110, align 4, !tbaa !58
  %111 = fmul float %83, 6.250000e-02
  %112 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %80
  store float %111, float* %112, align 4, !tbaa !58
  %113 = fmul float %89, 5.000000e-01
  %114 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %86
  store float %113, float* %114, align 4, !tbaa !58
  %115 = fmul float %95, 1.562500e-02
  %116 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %92
  store float %115, float* %116, align 4, !tbaa !58
  %117 = fmul float %101, 1.250000e-01
  %118 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %98
  store float %117, float* %118, align 4, !tbaa !58
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
  %140 = load bfloat, bfloat addrspace(1)* %139, align 2, !tbaa !56
  %141 = fpext bfloat %140 to float
  %142 = getelementptr inbounds bfloat, bfloat addrspace(1)* %138, i64 %123
  %143 = load bfloat, bfloat addrspace(1)* %142, align 2, !tbaa !56
  %144 = fpext bfloat %143 to float
  %145 = call fast float @_ZN22r5_raw_odd_control_tap23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %132, float* noundef nonnull %25, float noundef %141, float noundef %144, float noundef %103) #16
  %146 = zext i32 %126 to i64
  %147 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !58
  %149 = fadd float %145, %148
  store float %149, float* %147, align 4, !tbaa !58
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
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #14
  ret void

157:                                              ; preds = %182, %40
  %158 = phi i32 [ 0, %40 ], [ %183, %182 ]
  %159 = add i32 %158, %8
  %160 = icmp ult i32 %159, %41
  br i1 %160, label %161, label %182

161:                                              ; preds = %157
  %162 = zext i32 %158 to i64
  %163 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %162
  %164 = load float, float* %163, align 4, !tbaa !58
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
  store bfloat %167, bfloat addrspace(1)* %180, align 2, !tbaa !56
  %181 = getelementptr inbounds float, float addrspace(1)* %6, i64 %179
  store float %165, float addrspace(1)* %181, align 4, !tbaa !58
  br label %182

182:                                              ; preds = %178, %161, %157
  %183 = add nuw nsw i32 %158, 1
  %184 = icmp eq i32 %183, 4
  br i1 %184, label %156, label %157, !llvm.loop !83
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #5 {
  br label %4

3:                                                ; preds = %4
  ret float %54

4:                                                ; preds = %4, %2
  %5 = phi i1 [ true, %2 ], [ false, %4 ]
  %6 = phi i32 [ 0, %2 ], [ 8, %4 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %54, %4 ]
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %8
  %10 = load bfloat, bfloat addrspace(1)* %9, align 2, !tbaa !56
  %11 = fpext bfloat %10 to float
  %12 = or i32 %6, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %13
  %15 = load bfloat, bfloat addrspace(1)* %14, align 2, !tbaa !56
  %16 = fpext bfloat %15 to float
  %17 = fadd float %11, %16
  %18 = or i32 %6, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %19
  %21 = load bfloat, bfloat addrspace(1)* %20, align 2, !tbaa !56
  %22 = fpext bfloat %21 to float
  %23 = fadd float %17, %22
  %24 = or i32 %6, 3
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %25
  %27 = load bfloat, bfloat addrspace(1)* %26, align 2, !tbaa !56
  %28 = fpext bfloat %27 to float
  %29 = fadd float %23, %28
  %30 = or i32 %6, 4
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %31
  %33 = load bfloat, bfloat addrspace(1)* %32, align 2, !tbaa !56
  %34 = fpext bfloat %33 to float
  %35 = fadd float %29, %34
  %36 = or i32 %6, 5
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %37
  %39 = load bfloat, bfloat addrspace(1)* %38, align 2, !tbaa !56
  %40 = fpext bfloat %39 to float
  %41 = fadd float %35, %40
  %42 = or i32 %6, 6
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !56
  %46 = fpext bfloat %45 to float
  %47 = fadd float %41, %46
  %48 = or i32 %6, 7
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %49
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !56
  %52 = fpext bfloat %51 to float
  %53 = fadd float %47, %52
  %54 = fadd float %7, %53
  %55 = getelementptr inbounds float, float* %1, i64 %8
  store float %11, float* %55, align 4, !tbaa !58
  %56 = fmul float %16, 3.125000e-02
  %57 = getelementptr inbounds float, float* %1, i64 %13
  store float %56, float* %57, align 4, !tbaa !58
  %58 = fmul float %22, 2.500000e-01
  %59 = getelementptr inbounds float, float* %1, i64 %19
  store float %58, float* %59, align 4, !tbaa !58
  %60 = fmul float %28, 7.812500e-03
  %61 = getelementptr inbounds float, float* %1, i64 %25
  store float %60, float* %61, align 4, !tbaa !58
  %62 = fmul float %34, 6.250000e-02
  %63 = getelementptr inbounds float, float* %1, i64 %31
  store float %62, float* %63, align 4, !tbaa !58
  %64 = fmul float %40, 5.000000e-01
  %65 = getelementptr inbounds float, float* %1, i64 %37
  store float %64, float* %65, align 4, !tbaa !58
  %66 = fmul float %46, 1.562500e-02
  %67 = getelementptr inbounds float, float* %1, i64 %43
  store float %66, float* %67, align 4, !tbaa !58
  %68 = fmul float %52, 1.250000e-01
  %69 = getelementptr inbounds float, float* %1, i64 %49
  store float %68, float* %69, align 4, !tbaa !58
  br i1 %5, label %4, label %3, !llvm.loop !80
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #5 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %98, float %11)
  store float %12, float* %7, align 4, !tbaa !58
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %129, float %13)
  store float %14, float* %8, align 4, !tbaa !58
  ret void

15:                                               ; preds = %15, %9
  %16 = phi i8 addrspace(1)* [ %0, %9 ], [ %29, %15 ]
  %17 = phi float* [ %1, %9 ], [ %25, %15 ]
  %18 = phi float* [ %2, %9 ], [ %26, %15 ]
  %19 = phi float [ 0.000000e+00, %9 ], [ %98, %15 ]
  %20 = phi float [ 0.000000e+00, %9 ], [ %129, %15 ]
  %21 = phi i1 [ true, %9 ], [ false, %15 ]
  %22 = phi i32 [ 0, %9 ], [ 1, %15 ]
  %23 = shl nuw nsw i32 %22, 3
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds float, float* %17, i64 %24
  %26 = getelementptr inbounds float, float* %18, i64 %24
  %27 = mul nuw nsw i32 %22, 5
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds i8, i8 addrspace(1)* %16, i64 %28
  %30 = load i8, i8 addrspace(1)* %29, align 1, !tbaa !74
  %31 = zext i8 %30 to i32
  %32 = and i32 %31, 31
  %33 = and i32 %31, 224
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 1
  %35 = load i8, i8 addrspace(1)* %34, align 1, !tbaa !74
  %36 = zext i8 %35 to i32
  %37 = and i32 %36, 3
  %38 = and i32 %36, 124
  %39 = and i32 %36, 128
  %40 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 2
  %41 = load i8, i8 addrspace(1)* %40, align 1, !tbaa !74
  %42 = zext i8 %41 to i32
  %43 = and i32 %42, 15
  %44 = and i32 %42, 240
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 3
  %46 = load i8, i8 addrspace(1)* %45, align 1, !tbaa !74
  %47 = zext i8 %46 to i32
  %48 = and i32 %47, 1
  %49 = and i32 %47, 62
  %50 = and i32 %47, 192
  %51 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 4
  %52 = load i8, i8 addrspace(1)* %51, align 1, !tbaa !74
  %53 = zext i8 %52 to i32
  %54 = and i32 %53, 7
  %55 = and i32 %53, 248
  %56 = tail call float @air.convert.f.f32.s.i32(i32 %32) #10
  %57 = load float, float* %25, align 4, !tbaa !58
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %19)
  %59 = tail call float @air.convert.f.f32.s.i32(i32 %33) #10
  %60 = getelementptr inbounds float, float* %25, i64 1
  %61 = load float, float* %60, align 4, !tbaa !58
  %62 = tail call float @llvm.fmuladd.f32(float %59, float %61, float %58)
  %63 = tail call float @air.convert.f.f32.s.i32(i32 %37) #10
  %64 = fmul float %61, 2.560000e+02
  %65 = tail call float @llvm.fmuladd.f32(float %63, float %64, float %62)
  %66 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %67 = getelementptr inbounds float, float* %25, i64 2
  %68 = load float, float* %67, align 4, !tbaa !58
  %69 = tail call float @llvm.fmuladd.f32(float %66, float %68, float %65)
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %39) #10
  %71 = getelementptr inbounds float, float* %25, i64 3
  %72 = load float, float* %71, align 4, !tbaa !58
  %73 = tail call float @llvm.fmuladd.f32(float %70, float %72, float %69)
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %43) #10
  %75 = fmul float %72, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %74, float %75, float %73)
  %77 = tail call float @air.convert.f.f32.s.i32(i32 %44) #10
  %78 = getelementptr inbounds float, float* %25, i64 4
  %79 = load float, float* %78, align 4, !tbaa !58
  %80 = tail call float @llvm.fmuladd.f32(float %77, float %79, float %76)
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %48) #10
  %82 = fmul float %79, 2.560000e+02
  %83 = tail call float @llvm.fmuladd.f32(float %81, float %82, float %80)
  %84 = tail call float @air.convert.f.f32.s.i32(i32 %49) #10
  %85 = getelementptr inbounds float, float* %25, i64 5
  %86 = load float, float* %85, align 4, !tbaa !58
  %87 = tail call float @llvm.fmuladd.f32(float %84, float %86, float %83)
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %50) #10
  %89 = getelementptr inbounds float, float* %25, i64 6
  %90 = load float, float* %89, align 4, !tbaa !58
  %91 = tail call float @llvm.fmuladd.f32(float %88, float %90, float %87)
  %92 = tail call float @air.convert.f.f32.s.i32(i32 %54) #10
  %93 = fmul float %90, 2.560000e+02
  %94 = tail call float @llvm.fmuladd.f32(float %92, float %93, float %91)
  %95 = tail call float @air.convert.f.f32.s.i32(i32 %55) #10
  %96 = getelementptr inbounds float, float* %25, i64 7
  %97 = load float, float* %96, align 4, !tbaa !58
  %98 = tail call float @llvm.fmuladd.f32(float %95, float %97, float %94)
  %99 = load float, float* %26, align 4, !tbaa !58
  %100 = tail call float @llvm.fmuladd.f32(float %56, float %99, float %20)
  %101 = getelementptr inbounds float, float* %26, i64 1
  %102 = load float, float* %101, align 4, !tbaa !58
  %103 = tail call float @llvm.fmuladd.f32(float %59, float %102, float %100)
  %104 = fmul float %102, 2.560000e+02
  %105 = tail call float @llvm.fmuladd.f32(float %63, float %104, float %103)
  %106 = getelementptr inbounds float, float* %26, i64 2
  %107 = load float, float* %106, align 4, !tbaa !58
  %108 = tail call float @llvm.fmuladd.f32(float %66, float %107, float %105)
  %109 = getelementptr inbounds float, float* %26, i64 3
  %110 = load float, float* %109, align 4, !tbaa !58
  %111 = tail call float @llvm.fmuladd.f32(float %70, float %110, float %108)
  %112 = fmul float %110, 2.560000e+02
  %113 = tail call float @llvm.fmuladd.f32(float %74, float %112, float %111)
  %114 = getelementptr inbounds float, float* %26, i64 4
  %115 = load float, float* %114, align 4, !tbaa !58
  %116 = tail call float @llvm.fmuladd.f32(float %77, float %115, float %113)
  %117 = fmul float %115, 2.560000e+02
  %118 = tail call float @llvm.fmuladd.f32(float %81, float %117, float %116)
  %119 = getelementptr inbounds float, float* %26, i64 5
  %120 = load float, float* %119, align 4, !tbaa !58
  %121 = tail call float @llvm.fmuladd.f32(float %84, float %120, float %118)
  %122 = getelementptr inbounds float, float* %26, i64 6
  %123 = load float, float* %122, align 4, !tbaa !58
  %124 = tail call float @llvm.fmuladd.f32(float %88, float %123, float %121)
  %125 = fmul float %123, 2.560000e+02
  %126 = tail call float @llvm.fmuladd.f32(float %92, float %125, float %124)
  %127 = getelementptr inbounds float, float* %26, i64 7
  %128 = load float, float* %127, align 4, !tbaa !58
  %129 = tail call float @llvm.fmuladd.f32(float %95, float %128, float %126)
  br i1 %21, label %15, label %10, !llvm.loop !84
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4) local_unnamed_addr #5 {
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
  %21 = load i8, i8 addrspace(1)* %20, align 1, !tbaa !74
  %22 = zext i8 %21 to i32
  %23 = and i32 %22, 31
  %24 = tail call float @air.convert.f.f32.s.i32(i32 %23) #10
  %25 = load float, float* %17, align 4, !tbaa !58
  %26 = tail call float @llvm.fmuladd.f32(float %24, float %25, float %12)
  %27 = and i32 %22, 224
  %28 = tail call float @air.convert.f.f32.s.i32(i32 %27) #10
  %29 = getelementptr inbounds float, float* %17, i64 1
  %30 = load float, float* %29, align 4, !tbaa !58
  %31 = tail call float @llvm.fmuladd.f32(float %28, float %30, float %26)
  %32 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 1
  %33 = load i8, i8 addrspace(1)* %32, align 1, !tbaa !74
  %34 = zext i8 %33 to i32
  %35 = and i32 %34, 3
  %36 = tail call float @air.convert.f.f32.s.i32(i32 %35) #10
  %37 = fmul float %30, 2.560000e+02
  %38 = tail call float @llvm.fmuladd.f32(float %36, float %37, float %31)
  %39 = and i32 %34, 124
  %40 = tail call float @air.convert.f.f32.s.i32(i32 %39) #10
  %41 = getelementptr inbounds float, float* %17, i64 2
  %42 = load float, float* %41, align 4, !tbaa !58
  %43 = tail call float @llvm.fmuladd.f32(float %40, float %42, float %38)
  %44 = and i32 %34, 128
  %45 = tail call float @air.convert.f.f32.s.i32(i32 %44) #10
  %46 = getelementptr inbounds float, float* %17, i64 3
  %47 = load float, float* %46, align 4, !tbaa !58
  %48 = tail call float @llvm.fmuladd.f32(float %45, float %47, float %43)
  %49 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 2
  %50 = load i8, i8 addrspace(1)* %49, align 1, !tbaa !74
  %51 = zext i8 %50 to i32
  %52 = and i32 %51, 15
  %53 = tail call float @air.convert.f.f32.s.i32(i32 %52) #10
  %54 = fmul float %47, 2.560000e+02
  %55 = tail call float @llvm.fmuladd.f32(float %53, float %54, float %48)
  %56 = and i32 %51, 240
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %56) #10
  %58 = getelementptr inbounds float, float* %17, i64 4
  %59 = load float, float* %58, align 4, !tbaa !58
  %60 = tail call float @llvm.fmuladd.f32(float %57, float %59, float %55)
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 3
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !74
  %63 = zext i8 %62 to i32
  %64 = and i32 %63, 1
  %65 = tail call float @air.convert.f.f32.s.i32(i32 %64) #10
  %66 = fmul float %59, 2.560000e+02
  %67 = tail call float @llvm.fmuladd.f32(float %65, float %66, float %60)
  %68 = and i32 %63, 62
  %69 = tail call float @air.convert.f.f32.s.i32(i32 %68) #10
  %70 = getelementptr inbounds float, float* %17, i64 5
  %71 = load float, float* %70, align 4, !tbaa !58
  %72 = tail call float @llvm.fmuladd.f32(float %69, float %71, float %67)
  %73 = and i32 %63, 192
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %73) #10
  %75 = getelementptr inbounds float, float* %17, i64 6
  %76 = load float, float* %75, align 4, !tbaa !58
  %77 = tail call float @llvm.fmuladd.f32(float %74, float %76, float %72)
  %78 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 4
  %79 = load i8, i8 addrspace(1)* %78, align 1, !tbaa !74
  %80 = zext i8 %79 to i32
  %81 = and i32 %80, 7
  %82 = tail call float @air.convert.f.f32.s.i32(i32 %81) #10
  %83 = fmul float %76, 2.560000e+02
  %84 = tail call float @llvm.fmuladd.f32(float %82, float %83, float %77)
  %85 = and i32 %80, 248
  %86 = tail call float @air.convert.f.f32.s.i32(i32 %85) #10
  %87 = getelementptr inbounds float, float* %17, i64 7
  %88 = load float, float* %87, align 4, !tbaa !58
  %89 = tail call float @llvm.fmuladd.f32(float %86, float %88, float %84)
  br i1 %10, label %9, label %6, !llvm.loop !85
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt5ELt128EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !38
  switch i32 %3, label %4 [
    i32 6144, label %11
    i32 2560, label %7
  ]

4:                                                ; preds = %1
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !44
  br label %31

7:                                                ; preds = %1
  %8 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %9 = load i32, i32 addrspace(2)* %8, align 4, !tbaa !44
  %10 = icmp eq i32 %9, 6144
  br i1 %10, label %23, label %31

11:                                               ; preds = %1
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %13 = load i32, i32 addrspace(2)* %12, align 4, !tbaa !44
  %14 = icmp eq i32 %13, 2560
  br i1 %14, label %15, label %31

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %17 = load i64, i64 addrspace(2)* %16, align 8, !tbaa !51
  %18 = icmp ult i64 %17, 7208575253501192
  %19 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %20 = load i64, i64 addrspace(2)* %19, align 8
  %21 = icmp ult i64 %20, 7208575253501193
  %22 = select i1 %18, i1 %21, i1 false
  br label %54

23:                                               ; preds = %7
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8, !tbaa !51
  %26 = icmp ult i64 %25, 3002888502964277
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %28 = load i64, i64 addrspace(2)* %27, align 8
  %29 = icmp ult i64 %28, 3002888502964277
  %30 = select i1 %26, i1 %29, i1 false
  br label %54

31:                                               ; preds = %11, %7, %4
  %32 = phi i32 [ %6, %4 ], [ %9, %7 ], [ %13, %11 ]
  %33 = icmp ult i32 %32, 2
  br i1 %33, label %54, label %34

34:                                               ; preds = %31
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %36 = load i64, i64 addrspace(2)* %35, align 8, !tbaa !51
  %37 = zext i32 %3 to i64
  %38 = mul nuw nsw i64 %37, 5
  %39 = lshr i64 %38, 3
  %40 = xor i64 %39, -1
  %41 = add i32 %32, -1
  %42 = zext i32 %41 to i64
  %43 = udiv i64 %40, %42
  %44 = icmp ugt i64 %36, %43
  br i1 %44, label %54, label %45

45:                                               ; preds = %34
  %46 = lshr i32 %3, 6
  %47 = and i32 %46, 67108862
  %48 = zext i32 %47 to i64
  %49 = xor i64 %48, -1
  %50 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %51 = load i64, i64 addrspace(2)* %50, align 8, !tbaa !52
  %52 = udiv i64 %49, %42
  %53 = icmp ule i64 %51, %52
  br label %54

54:                                               ; preds = %45, %34, %31, %23, %15
  %55 = phi i1 [ %22, %15 ], [ %30, %23 ], [ false, %31 ], [ false, %34 ], [ %53, %45 ]
  ret i1 %55
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #6 {
  %13 = alloca [16 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %15) #14
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !38
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %20, label %23

20:                                               ; preds = %12
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !44
  br label %40

23:                                               ; preds = %12
  %24 = shl i32 %11, 4
  %25 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %26 = zext i32 %9 to i64
  %27 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 11
  %28 = load i64, i64 addrspace(2)* %27, align 8, !tbaa !53
  %29 = mul i64 %28, %26
  %30 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 9
  %31 = load i64, i64 addrspace(2)* %30, align 8, !tbaa !76
  %32 = mul i64 %31, %26
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %34 = load i32, i32 addrspace(2)* %33, align 4, !tbaa !44
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
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !56
  %60 = fpext bfloat %59 to float
  %61 = or i32 %55, 1
  %62 = zext i32 %61 to i64
  %63 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %62
  %64 = load bfloat, bfloat addrspace(1)* %63, align 2, !tbaa !56
  %65 = fpext bfloat %64 to float
  %66 = fadd float %60, %65
  %67 = or i32 %55, 2
  %68 = zext i32 %67 to i64
  %69 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %68
  %70 = load bfloat, bfloat addrspace(1)* %69, align 2, !tbaa !56
  %71 = fpext bfloat %70 to float
  %72 = fadd float %66, %71
  %73 = or i32 %55, 3
  %74 = zext i32 %73 to i64
  %75 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %74
  %76 = load bfloat, bfloat addrspace(1)* %75, align 2, !tbaa !56
  %77 = fpext bfloat %76 to float
  %78 = fadd float %72, %77
  %79 = or i32 %55, 4
  %80 = zext i32 %79 to i64
  %81 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %80
  %82 = load bfloat, bfloat addrspace(1)* %81, align 2, !tbaa !56
  %83 = fpext bfloat %82 to float
  %84 = fadd float %78, %83
  %85 = or i32 %55, 5
  %86 = zext i32 %85 to i64
  %87 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %86
  %88 = load bfloat, bfloat addrspace(1)* %87, align 2, !tbaa !56
  %89 = fpext bfloat %88 to float
  %90 = fadd float %84, %89
  %91 = or i32 %55, 6
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %92
  %94 = load bfloat, bfloat addrspace(1)* %93, align 2, !tbaa !56
  %95 = fpext bfloat %94 to float
  %96 = fadd float %90, %95
  %97 = or i32 %55, 7
  %98 = zext i32 %97 to i64
  %99 = getelementptr inbounds bfloat, bfloat addrspace(1)* %52, i64 %98
  %100 = load bfloat, bfloat addrspace(1)* %99, align 2, !tbaa !56
  %101 = fpext bfloat %100 to float
  %102 = fadd float %96, %101
  %103 = fadd float %56, %102
  %104 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %57
  store float %60, float* %104, align 4, !tbaa !58
  %105 = fmul float %65, 3.125000e-02
  %106 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %62
  store float %105, float* %106, align 4, !tbaa !58
  %107 = fmul float %71, 2.500000e-01
  %108 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %68
  store float %107, float* %108, align 4, !tbaa !58
  %109 = fmul float %77, 7.812500e-03
  %110 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %74
  store float %109, float* %110, align 4, !tbaa !58
  %111 = fmul float %83, 6.250000e-02
  %112 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %80
  store float %111, float* %112, align 4, !tbaa !58
  %113 = fmul float %89, 5.000000e-01
  %114 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %86
  store float %113, float* %114, align 4, !tbaa !58
  %115 = fmul float %95, 1.562500e-02
  %116 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %92
  store float %115, float* %116, align 4, !tbaa !58
  %117 = fmul float %101, 1.250000e-01
  %118 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 %98
  store float %117, float* %118, align 4, !tbaa !58
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
  %140 = load bfloat, bfloat addrspace(1)* %139, align 2, !tbaa !56
  %141 = fpext bfloat %140 to float
  %142 = getelementptr inbounds bfloat, bfloat addrspace(1)* %138, i64 %123
  %143 = load bfloat, bfloat addrspace(1)* %142, align 2, !tbaa !56
  %144 = fpext bfloat %143 to float
  %145 = call fast float @_ZN22r5_raw_odd_control_tap23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %132, float* noundef nonnull %25, float noundef %141, float noundef %144, float noundef %103) #16
  %146 = zext i32 %126 to i64
  %147 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %146
  %148 = load float, float* %147, align 4, !tbaa !58
  %149 = fadd float %145, %148
  store float %149, float* %147, align 4, !tbaa !58
  br label %150

150:                                              ; preds = %129, %125
  %151 = add nuw nsw i32 %126, 1
  %152 = icmp eq i32 %151, 4
  br i1 %152, label %153, label %125, !llvm.loop !86

153:                                              ; preds = %150
  %154 = add i32 %49, 512
  %155 = icmp ult i32 %154, %18
  br i1 %155, label %48, label %40, !llvm.loop !87

156:                                              ; preds = %182
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %15) #14
  ret void

157:                                              ; preds = %182, %40
  %158 = phi i32 [ 0, %40 ], [ %183, %182 ]
  %159 = add i32 %158, %8
  %160 = icmp ult i32 %159, %41
  br i1 %160, label %161, label %182

161:                                              ; preds = %157
  %162 = zext i32 %158 to i64
  %163 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %162
  %164 = load float, float* %163, align 4, !tbaa !58
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
  store bfloat %167, bfloat addrspace(1)* %180, align 2, !tbaa !56
  %181 = getelementptr inbounds float, float addrspace(1)* %6, i64 %179
  store float %165, float addrspace(1)* %181, align 4, !tbaa !58
  br label %182

182:                                              ; preds = %178, %161, %157
  %183 = add nuw nsw i32 %158, 1
  %184 = icmp eq i32 %183, 4
  br i1 %184, label %156, label %157, !llvm.loop !88
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt6ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !38
  %4 = icmp eq i32 %3, 2560
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !44
  %7 = icmp eq i32 %6, 6144
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %9, label %17

9:                                                ; preds = %1
  %10 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %11 = load i64, i64 addrspace(2)* %10, align 8, !tbaa !51
  %12 = icmp ult i64 %11, 3002888502964277
  %13 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %14 = load i64, i64 addrspace(2)* %13, align 8
  %15 = icmp ult i64 %14, 3002888502964277
  %16 = select i1 %12, i1 %15, i1 false
  br label %39

17:                                               ; preds = %1
  %18 = icmp ult i32 %6, 2
  br i1 %18, label %39, label %19

19:                                               ; preds = %17
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %21 = load i64, i64 addrspace(2)* %20, align 8, !tbaa !51
  %22 = zext i32 %3 to i64
  %23 = mul nuw nsw i64 %22, 6
  %24 = lshr i64 %23, 3
  %25 = xor i64 %24, -1
  %26 = add i32 %6, -1
  %27 = zext i32 %26 to i64
  %28 = udiv i64 %25, %27
  %29 = icmp ugt i64 %21, %28
  br i1 %29, label %39, label %30

30:                                               ; preds = %19
  %31 = lshr i32 %3, 5
  %32 = and i32 %31, 134217726
  %33 = zext i32 %32 to i64
  %34 = xor i64 %33, -1
  %35 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %36 = load i64, i64 addrspace(2)* %35, align 8, !tbaa !52
  %37 = udiv i64 %34, %27
  %38 = icmp ule i64 %36, %37
  br label %39

39:                                               ; preds = %30, %19, %17, %9
  %40 = phi i1 [ %16, %9 ], [ false, %17 ], [ false, %19 ], [ %38, %30 ]
  ret i1 %40
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap12project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, float addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %9, i64 noundef %10, i32 noundef %11) local_unnamed_addr #6 {
  %13 = alloca [8 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %15) #14
  %16 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %16, i8 0, i64 16, i1 false)
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !38
  %19 = icmp eq i32 %18, 0
  br i1 %19, label %24, label %20

20:                                               ; preds = %12
  %21 = shl i32 %11, 3
  %22 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %23 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 0
  br label %33

24:                                               ; preds = %72, %12
  %25 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %26 = load i32, i32 addrspace(2)* %25, align 4, !tbaa !44
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
  %44 = load bfloat, bfloat addrspace(1)* %43, align 2, !tbaa !56
  %45 = fpext bfloat %44 to float
  %46 = or i32 %40, 1
  %47 = zext i32 %46 to i64
  %48 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %47
  %49 = load bfloat, bfloat addrspace(1)* %48, align 2, !tbaa !56
  %50 = fpext bfloat %49 to float
  %51 = fadd float %45, %50
  %52 = or i32 %40, 2
  %53 = zext i32 %52 to i64
  %54 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %53
  %55 = load bfloat, bfloat addrspace(1)* %54, align 2, !tbaa !56
  %56 = fpext bfloat %55 to float
  %57 = fadd float %51, %56
  %58 = or i32 %40, 3
  %59 = zext i32 %58 to i64
  %60 = getelementptr inbounds bfloat, bfloat addrspace(1)* %37, i64 %59
  %61 = load bfloat, bfloat addrspace(1)* %60, align 2, !tbaa !56
  %62 = fpext bfloat %61 to float
  %63 = fadd float %57, %62
  %64 = fadd float %41, %63
  %65 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %42
  store float %45, float* %65, align 4, !tbaa !58
  %66 = fmul float %50, 1.562500e-02
  %67 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %47
  store float %66, float* %67, align 4, !tbaa !58
  %68 = fmul float %56, 6.250000e-02
  %69 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %53
  store float %68, float* %69, align 4, !tbaa !58
  %70 = fmul float %62, 2.500000e-01
  %71 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 %59
  store float %70, float* %71, align 4, !tbaa !58
  br i1 %39, label %38, label %72, !llvm.loop !89

72:                                               ; preds = %38
  call void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, i32 noundef %8, i32 noundef %35, i32 noundef %9, float* noundef nonnull %22, float noundef %64, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %23) #12
  %73 = add i32 %34, 256
  %74 = icmp ult i32 %73, %18
  br i1 %74, label %33, label %24, !llvm.loop !90

75:                                               ; preds = %101
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %16) #14
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %15) #14
  ret void

76:                                               ; preds = %101, %24
  %77 = phi i32 [ 0, %24 ], [ %102, %101 ]
  %78 = add i32 %77, %8
  %79 = icmp ult i32 %78, %26
  br i1 %79, label %80, label %101

80:                                               ; preds = %76
  %81 = zext i32 %77 to i64
  %82 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %81
  %83 = load float, float* %82, align 4, !tbaa !58
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
  store bfloat %86, bfloat addrspace(1)* %99, align 2, !tbaa !56
  %100 = getelementptr inbounds float, float addrspace(1)* %6, i64 %98
  store float %84, float addrspace(1)* %100, align 4, !tbaa !58
  br label %101

101:                                              ; preds = %97, %80, %76
  %102 = add nuw nsw i32 %77, 1
  %103 = icmp eq i32 %102, 4
  br i1 %103, label %75, label %76, !llvm.loop !91
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #5 {
  br label %4

3:                                                ; preds = %4
  ret float %30

4:                                                ; preds = %4, %2
  %5 = phi i1 [ true, %2 ], [ false, %4 ]
  %6 = phi i32 [ 0, %2 ], [ 4, %4 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %30, %4 ]
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %8
  %10 = load bfloat, bfloat addrspace(1)* %9, align 2, !tbaa !56
  %11 = fpext bfloat %10 to float
  %12 = or i32 %6, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %13
  %15 = load bfloat, bfloat addrspace(1)* %14, align 2, !tbaa !56
  %16 = fpext bfloat %15 to float
  %17 = fadd float %11, %16
  %18 = or i32 %6, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %19
  %21 = load bfloat, bfloat addrspace(1)* %20, align 2, !tbaa !56
  %22 = fpext bfloat %21 to float
  %23 = fadd float %17, %22
  %24 = or i32 %6, 3
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %25
  %27 = load bfloat, bfloat addrspace(1)* %26, align 2, !tbaa !56
  %28 = fpext bfloat %27 to float
  %29 = fadd float %23, %28
  %30 = fadd float %7, %29
  %31 = getelementptr inbounds float, float* %1, i64 %8
  store float %11, float* %31, align 4, !tbaa !58
  %32 = fmul float %16, 1.562500e-02
  %33 = getelementptr inbounds float, float* %1, i64 %13
  store float %32, float* %33, align 4, !tbaa !58
  %34 = fmul float %22, 6.250000e-02
  %35 = getelementptr inbounds float, float* %1, i64 %19
  store float %34, float* %35, align 4, !tbaa !58
  %36 = fmul float %28, 2.500000e-01
  %37 = getelementptr inbounds float, float* %1, i64 %25
  store float %36, float* %37, align 4, !tbaa !58
  br i1 %5, label %4, label %3, !llvm.loop !89
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #5 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %64, float %11)
  store float %12, float* %7, align 4, !tbaa !58
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %79, float %13)
  store float %14, float* %8, align 4, !tbaa !58
  ret void

15:                                               ; preds = %15, %9
  %16 = phi i8 addrspace(1)* [ %0, %9 ], [ %29, %15 ]
  %17 = phi float* [ %1, %9 ], [ %25, %15 ]
  %18 = phi float* [ %2, %9 ], [ %26, %15 ]
  %19 = phi float [ 0.000000e+00, %9 ], [ %64, %15 ]
  %20 = phi float [ 0.000000e+00, %9 ], [ %79, %15 ]
  %21 = phi i1 [ true, %9 ], [ false, %15 ]
  %22 = phi i32 [ 0, %9 ], [ 1, %15 ]
  %23 = shl nuw nsw i32 %22, 2
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds float, float* %17, i64 %24
  %26 = getelementptr inbounds float, float* %18, i64 %24
  %27 = mul nuw nsw i32 %22, 3
  %28 = zext i32 %27 to i64
  %29 = getelementptr inbounds i8, i8 addrspace(1)* %16, i64 %28
  %30 = load i8, i8 addrspace(1)* %29, align 1, !tbaa !74
  %31 = zext i8 %30 to i32
  %32 = and i32 %31, 63
  %33 = and i32 %31, 192
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 1
  %35 = load i8, i8 addrspace(1)* %34, align 1, !tbaa !74
  %36 = zext i8 %35 to i32
  %37 = and i32 %36, 15
  %38 = and i32 %36, 240
  %39 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 2
  %40 = load i8, i8 addrspace(1)* %39, align 1, !tbaa !74
  %41 = zext i8 %40 to i32
  %42 = and i32 %41, 3
  %43 = and i32 %41, 252
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %32) #10
  %45 = load float, float* %25, align 4, !tbaa !58
  %46 = tail call float @llvm.fmuladd.f32(float %44, float %45, float %19)
  %47 = tail call float @air.convert.f.f32.s.i32(i32 %33) #10
  %48 = getelementptr inbounds float, float* %25, i64 1
  %49 = load float, float* %48, align 4, !tbaa !58
  %50 = tail call float @llvm.fmuladd.f32(float %47, float %49, float %46)
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %37) #10
  %52 = fmul float %49, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %50)
  %54 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %55 = getelementptr inbounds float, float* %25, i64 2
  %56 = load float, float* %55, align 4, !tbaa !58
  %57 = tail call float @llvm.fmuladd.f32(float %54, float %56, float %53)
  %58 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %59 = fmul float %56, 2.560000e+02
  %60 = tail call float @llvm.fmuladd.f32(float %58, float %59, float %57)
  %61 = tail call float @air.convert.f.f32.s.i32(i32 %43) #10
  %62 = getelementptr inbounds float, float* %25, i64 3
  %63 = load float, float* %62, align 4, !tbaa !58
  %64 = tail call float @llvm.fmuladd.f32(float %61, float %63, float %60)
  %65 = load float, float* %26, align 4, !tbaa !58
  %66 = tail call float @llvm.fmuladd.f32(float %44, float %65, float %20)
  %67 = getelementptr inbounds float, float* %26, i64 1
  %68 = load float, float* %67, align 4, !tbaa !58
  %69 = tail call float @llvm.fmuladd.f32(float %47, float %68, float %66)
  %70 = fmul float %68, 2.560000e+02
  %71 = tail call float @llvm.fmuladd.f32(float %51, float %70, float %69)
  %72 = getelementptr inbounds float, float* %26, i64 2
  %73 = load float, float* %72, align 4, !tbaa !58
  %74 = tail call float @llvm.fmuladd.f32(float %54, float %73, float %71)
  %75 = fmul float %73, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %58, float %75, float %74)
  %77 = getelementptr inbounds float, float* %26, i64 3
  %78 = load float, float* %77, align 4, !tbaa !58
  %79 = tail call float @llvm.fmuladd.f32(float %61, float %78, float %76)
  br i1 %21, label %15, label %10, !llvm.loop !92
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN22r5_raw_odd_control_tap16accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #6 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !53
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !76
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !44
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
  %55 = tail call fast float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %41, float* noundef %7, float noundef %50, float noundef %53, float noundef %8, i32 noundef %9) #13
  %56 = zext i32 %35 to i64
  %57 = getelementptr inbounds float, float* %11, i64 %56
  %58 = load float, float* %57, align 4, !tbaa !58
  %59 = fadd float %55, %58
  store float %59, float* %57, align 4, !tbaa !58
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
  %72 = load i8, i8 addrspace(1)* %71, align 1, !tbaa !74
  %73 = zext i8 %72 to i32
  %74 = and i32 %73, 63
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = load float, float* %68, align 4, !tbaa !58
  %77 = tail call float @llvm.fmuladd.f32(float %75, float %76, float %63) #14
  %78 = and i32 %73, 192
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #10
  %80 = getelementptr inbounds float, float* %68, i64 1
  %81 = load float, float* %80, align 4, !tbaa !58
  %82 = tail call float @llvm.fmuladd.f32(float %79, float %81, float %77) #14
  %83 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 1
  %84 = load i8, i8 addrspace(1)* %83, align 1, !tbaa !74
  %85 = zext i8 %84 to i32
  %86 = and i32 %85, 15
  %87 = tail call float @air.convert.f.f32.s.i32(i32 %86) #10
  %88 = fmul float %81, 2.560000e+02
  %89 = tail call float @llvm.fmuladd.f32(float %87, float %88, float %82) #14
  %90 = and i32 %85, 240
  %91 = tail call float @air.convert.f.f32.s.i32(i32 %90) #10
  %92 = getelementptr inbounds float, float* %68, i64 2
  %93 = load float, float* %92, align 4, !tbaa !58
  %94 = tail call float @llvm.fmuladd.f32(float %91, float %93, float %89) #14
  %95 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 2
  %96 = load i8, i8 addrspace(1)* %95, align 1, !tbaa !74
  %97 = zext i8 %96 to i32
  %98 = and i32 %97, 3
  %99 = tail call float @air.convert.f.f32.s.i32(i32 %98) #10
  %100 = fmul float %93, 2.560000e+02
  %101 = tail call float @llvm.fmuladd.f32(float %99, float %100, float %94) #14
  %102 = and i32 %97, 252
  %103 = tail call float @air.convert.f.f32.s.i32(i32 %102) #10
  %104 = getelementptr inbounds float, float* %68, i64 3
  %105 = load float, float* %104, align 4, !tbaa !58
  %106 = tail call float @llvm.fmuladd.f32(float %103, float %105, float %101) #14
  br i1 %61, label %60, label %107, !llvm.loop !93

107:                                              ; preds = %60
  %108 = fmul float %53, %8
  %109 = tail call float @llvm.fmuladd.f32(float %50, float %106, float %108) #14
  %110 = zext i32 %35 to i64
  %111 = getelementptr inbounds float, float* %11, i64 %110
  %112 = load float, float* %111, align 4, !tbaa !58
  %113 = fadd float %112, %109
  store float %113, float* %111, align 4, !tbaa !58
  br label %114

114:                                              ; preds = %107, %54, %34
  %115 = add nuw nsw i32 %35, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %33, label %34, !llvm.loop !94
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN22r5_raw_odd_control_tap28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #5 {
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
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !74
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 63
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #10
  %28 = load float, float* %20, align 4, !tbaa !58
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 192
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #10
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !58
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !74
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 15
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 240
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !58
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %48 = load i8, i8 addrspace(1)* %47, align 1, !tbaa !74
  %49 = zext i8 %48 to i32
  %50 = and i32 %49, 3
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %50) #10
  %52 = fmul float %45, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %46)
  %54 = and i32 %49, 252
  %55 = tail call float @air.convert.f.f32.s.i32(i32 %54) #10
  %56 = getelementptr inbounds float, float* %20, i64 3
  %57 = load float, float* %56, align 4, !tbaa !58
  %58 = tail call float @llvm.fmuladd.f32(float %55, float %57, float %53)
  %59 = add nuw nsw i32 %14, 1
  %60 = icmp eq i32 %59, %7
  br i1 %60, label %9, label %13, !llvm.loop !95
}

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #1 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #2 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #3 = { mustprogress nounwind willreturn }
attributes #4 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #5 = { inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #6 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #7 = { argmemonly nofree nounwind willreturn writeonly }
attributes #8 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #9 = { convergent mustprogress nounwind willreturn }
attributes #10 = { nounwind readnone willreturn }
attributes #11 = { nounwind willreturn }
attributes #12 = { convergent nobuiltin "no-builtins" }
attributes #13 = { nobuiltin "no-builtins" }
attributes #14 = { nounwind }
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
!9 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @r5_raw_odd_rowpair_sep22_candidate_probe_q4_g64, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !20, !22, !23, !24, !25, !26, !27}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"input"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"weights"}
!14 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"scales"}
!15 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"biases"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"long", !"air.arg_name", !"expert_ids", !"air.arg_unused"}
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
!28 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @r5_raw_odd_rowpair_sep22_candidate_probe_q5_g64, !10, !11}
!29 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @r5_raw_odd_rowpair_sep22_candidate_probe_q5_g128, !10, !11}
!30 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @r5_raw_odd_rowpair_sep22_candidate_probe_q6_g64, !10, !11}
!31 = !{!"air.compile.denorms_disable"}
!32 = !{!"air.compile.fast_math_enable"}
!33 = !{!"air.compile.framebuffer_fetch_enable"}
!34 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!35 = !{i32 2, i32 9, i32 0}
!36 = !{!"Metal", i32 4, i32 1, i32 0}
!37 = !{!"/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_guard_specialized_rowpair_sep22/kernel/candidate_probe.metal"}
!38 = !{!39, !40, i64 8}
!39 = !{!"_ZTS17FlashAffineParams", !40, i64 0, !40, i64 4, !40, i64 8, !40, i64 12, !40, i64 16, !40, i64 20, !40, i64 24, !40, i64 28, !43, i64 32, !43, i64 40, !43, i64 48, !43, i64 56}
!40 = !{!"int", !41, i64 0}
!41 = !{!"omnipotent char", !42, i64 0}
!42 = !{!"Simple C++ TBAA"}
!43 = !{!"long", !41, i64 0}
!44 = !{!39, !40, i64 12}
!45 = !{!39, !40, i64 0}
!46 = !{!39, !40, i64 4}
!47 = !{!39, !40, i64 16}
!48 = !{!39, !40, i64 28}
!49 = !{!39, !40, i64 20}
!50 = !{!39, !40, i64 24}
!51 = !{!39, !43, i64 32}
!52 = !{!39, !43, i64 48}
!53 = !{!39, !43, i64 56}
!54 = distinct !{!54, !55}
!55 = !{!"llvm.loop.mustprogress"}
!56 = !{!57, !57, i64 0}
!57 = !{!"bfloat", !41, i64 0}
!58 = !{!59, !59, i64 0}
!59 = !{!"float", !41, i64 0}
!60 = distinct !{!60, !55}
!61 = distinct !{!61, !55}
!62 = distinct !{!62, !55}
!63 = distinct !{!63, !55}
!64 = distinct !{!64, !55}
!65 = distinct !{!65, !55}
!66 = distinct !{!66, !55}
!67 = distinct !{!67, !55}
!68 = distinct !{!68, !55}
!69 = distinct !{!69, !55}
!70 = distinct !{!70, !55}
!71 = distinct !{!71, !55}
!72 = distinct !{!72, !55}
!73 = distinct !{!73, !55}
!74 = !{!41, !41, i64 0}
!75 = distinct !{!75, !55}
!76 = !{!39, !43, i64 40}
!77 = distinct !{!77, !55}
!78 = distinct !{!78, !55}
!79 = distinct !{!79, !55}
!80 = distinct !{!80, !55}
!81 = distinct !{!81, !55}
!82 = distinct !{!82, !55}
!83 = distinct !{!83, !55}
!84 = distinct !{!84, !55}
!85 = distinct !{!85, !55}
!86 = distinct !{!86, !55}
!87 = distinct !{!87, !55}
!88 = distinct !{!88, !55}
!89 = distinct !{!89, !55}
!90 = distinct !{!90, !55}
!91 = distinct !{!91, !55}
!92 = distinct !{!92, !55}
!93 = distinct !{!93, !55}
!94 = distinct !{!94, !55}
!95 = distinct !{!95, !55}
