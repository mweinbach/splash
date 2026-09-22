; ModuleID = '/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/_cpu_build_v1/candidate_probe.air'
source_filename = "/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/candidate_probe.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

%"struct.metal::_atomic" = type { i32 }
%struct.FlashAffineParams = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64 }

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_candidate_probe_q4_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #9
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
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #11
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
  %26 = icmp eq i32 %25, 4
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
  %34 = icmp eq i32 %33, 4
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
  %84 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt4ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #12
  br i1 %84, label %90, label %85

85:                                               ; preds = %83, %78, %68, %62, %52, %48, %44, %40, %35, %31, %28, %20, %11
  %86 = icmp eq i32 %10, 0
  br i1 %86, label %87, label %222

87:                                               ; preds = %85
  %88 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %89 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %88, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %222

90:                                               ; preds = %83
  %91 = extractelement <3 x i32> %8, i64 0
  %92 = shl i32 %91, 3
  %93 = shl i32 %9, 2
  %94 = add i32 %92, %93
  %95 = lshr i32 %36, 3
  %96 = icmp uge i32 %91, %95
  %97 = extractelement <3 x i32> %8, i64 1
  %98 = icmp ugt i32 %97, 1
  %99 = or i1 %98, %96
  %100 = extractelement <3 x i32> %8, i64 2
  %101 = icmp ne i32 %100, 0
  %102 = or i1 %101, %99
  %103 = xor i1 %102, true
  %104 = icmp ult i32 %94, %36
  %105 = select i1 %103, i1 %104, i1 false
  br i1 %105, label %106, label %222

106:                                              ; preds = %90
  %107 = zext i32 %97 to i64
  %108 = shl nuw nsw i64 %107, 1
  %109 = or i64 %108, 1
  %110 = zext i32 %19 to i64
  %111 = mul nuw nsw i64 %108, %110
  %112 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %111
  %113 = mul i64 %109, %110
  %114 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %113
  %115 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %115) #13
  %116 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %116) #13
  %117 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %117) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %117, i8 0, i64 16, i1 false)
  %118 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %118) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %118, i8 0, i64 16, i1 false)
  %119 = shl i32 %10, 4
  %120 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %121 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %122 = bitcast float* %16 to i8*
  %123 = bitcast float* %17 to i8*
  br label %133

124:                                              ; preds = %145
  %125 = icmp eq i32 %10, 0
  %126 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %127 = zext i32 %36 to i64
  %128 = mul i64 %108, %127
  %129 = zext i32 %94 to i64
  %130 = add i64 %128, %129
  %131 = mul i64 %109, %127
  %132 = add i64 %131, %129
  br label %177

133:                                              ; preds = %145, %106
  %134 = phi i32 [ 0, %106 ], [ %146, %145 ]
  %135 = add i32 %134, %119
  %136 = zext i32 %135 to i64
  %137 = getelementptr inbounds bfloat, bfloat addrspace(1)* %112, i64 %136
  %138 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %137, float* noundef nonnull %120) #12
  %139 = getelementptr inbounds bfloat, bfloat addrspace(1)* %114, i64 %136
  %140 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %139, float* noundef nonnull %121) #12
  %141 = lshr i32 %135, 6
  %142 = lshr exact i64 %136, 1
  %143 = zext i32 %141 to i64
  %144 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %142
  br label %148

145:                                              ; preds = %148
  %146 = add i32 %134, 512
  %147 = icmp ult i32 %146, %19
  br i1 %147, label %133, label %124, !llvm.loop !54

148:                                              ; preds = %148, %133
  %149 = phi i32 [ 0, %133 ], [ %174, %148 ]
  %150 = add nuw nsw i32 %149, %94
  %151 = zext i32 %150 to i64
  %152 = mul i64 %64, %151
  %153 = getelementptr inbounds i8, i8 addrspace(1)* %144, i64 %152
  %154 = mul i64 %70, %151
  %155 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %154
  %156 = bitcast i8 addrspace(1)* %155 to bfloat addrspace(1)*
  %157 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %154
  %158 = bitcast i8 addrspace(1)* %157 to bfloat addrspace(1)*
  %159 = getelementptr inbounds bfloat, bfloat addrspace(1)* %156, i64 %143
  %160 = load bfloat, bfloat addrspace(1)* %159, align 2, !tbaa !56
  %161 = fpext bfloat %160 to float
  %162 = getelementptr inbounds bfloat, bfloat addrspace(1)* %158, i64 %143
  %163 = load bfloat, bfloat addrspace(1)* %162, align 2, !tbaa !56
  %164 = fpext bfloat %163 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %122) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %123) #13
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %153, float* noundef nonnull %120, float* noundef nonnull %121, float noundef %161, float noundef %164, float noundef %138, float noundef %140, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #12
  %165 = load float, float* %16, align 4, !tbaa !58
  %166 = zext i32 %149 to i64
  %167 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %166
  %168 = load float, float* %167, align 4, !tbaa !58
  %169 = fadd float %165, %168
  store float %169, float* %167, align 4, !tbaa !58
  %170 = load float, float* %17, align 4, !tbaa !58
  %171 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %166
  %172 = load float, float* %171, align 4, !tbaa !58
  %173 = fadd float %170, %172
  store float %173, float* %171, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %123) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %122) #13
  %174 = add nuw nsw i32 %149, 1
  %175 = icmp eq i32 %174, 4
  br i1 %175, label %145, label %148, !llvm.loop !60

176:                                              ; preds = %219
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %118) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %117) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %116) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %115) #13
  br label %222

177:                                              ; preds = %219, %124
  %178 = phi i16 [ 0, %124 ], [ %220, %219 ]
  %179 = zext i16 %178 to i64
  %180 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %179
  %181 = load float, float* %180, align 4, !tbaa !58
  %182 = call fast float @air.simd_sum.f32(float %181) #14
  br i1 %125, label %183, label %199

183:                                              ; preds = %177
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
  %194 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %126, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %195

195:                                              ; preds = %193, %188
  %196 = add i64 %130, %179
  %197 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %196
  store bfloat %184, bfloat addrspace(1)* %197, align 2, !tbaa !56
  %198 = getelementptr inbounds float, float addrspace(1)* %6, i64 %196
  store float %182, float addrspace(1)* %198, align 4, !tbaa !58
  br label %199

199:                                              ; preds = %195, %177
  %200 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %179
  %201 = load float, float* %200, align 4, !tbaa !58
  %202 = call fast float @air.simd_sum.f32(float %201) #14
  br i1 %125, label %203, label %219

203:                                              ; preds = %199
  %204 = fptrunc float %202 to bfloat
  %205 = bitcast float %202 to i32
  %206 = and i32 %205, 2139095040
  %207 = icmp eq i32 %206, 2139095040
  br i1 %207, label %213, label %208

208:                                              ; preds = %203
  %209 = fpext bfloat %204 to float
  %210 = bitcast float %209 to i32
  %211 = and i32 %210, 2139095040
  %212 = icmp eq i32 %211, 2139095040
  br i1 %212, label %213, label %215

213:                                              ; preds = %208, %203
  %214 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %126, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %215

215:                                              ; preds = %213, %208
  %216 = add i64 %132, %179
  %217 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %216
  store bfloat %204, bfloat addrspace(1)* %217, align 2, !tbaa !56
  %218 = getelementptr inbounds float, float addrspace(1)* %6, i64 %216
  store float %202, float addrspace(1)* %218, align 4, !tbaa !58
  br label %219

219:                                              ; preds = %215, %199
  %220 = add nuw nsw i16 %178, 1
  %221 = icmp eq i16 %220, 4
  br i1 %221, label %176, label %177, !llvm.loop !61

222:                                              ; preds = %176, %90, %87, %85
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_candidate_probe_q5_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #9
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
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #11
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
  %27 = icmp eq i32 %26, 4
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
  %66 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt5ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #12
  br i1 %66, label %72, label %67

67:                                               ; preds = %65, %60, %53, %49, %45, %41, %37, %33, %29, %21, %11
  %68 = icmp eq i32 %10, 0
  br i1 %68, label %69, label %201

69:                                               ; preds = %67
  %70 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %71 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %70, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %201

72:                                               ; preds = %65
  %73 = extractelement <3 x i32> %8, i64 0
  %74 = shl i32 %73, 3
  %75 = shl i32 %9, 2
  %76 = add i32 %74, %75
  %77 = icmp ugt i32 %73, 1279
  %78 = extractelement <3 x i32> %8, i64 1
  %79 = icmp ugt i32 %78, 1
  %80 = or i1 %79, %77
  %81 = extractelement <3 x i32> %8, i64 2
  %82 = icmp ne i32 %81, 0
  %83 = or i1 %82, %80
  %84 = icmp ugt i32 %76, 10239
  %85 = or i1 %83, %84
  br i1 %85, label %201, label %86

86:                                               ; preds = %72
  %87 = zext i32 %78 to i64
  %88 = shl nuw nsw i64 %87, 1
  %89 = or i64 %88, 1
  %90 = mul nuw nsw i64 %87, 5120
  %91 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %90
  %92 = mul nuw nsw i64 %89, 2560
  %93 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %92
  %94 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %94) #13
  %95 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %95) #13
  %96 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %96, i8 0, i64 16, i1 false)
  %97 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %97) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %97, i8 0, i64 16, i1 false)
  %98 = shl i32 %10, 4
  %99 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %100 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %101 = bitcast float* %16 to i8*
  %102 = bitcast float* %17 to i8*
  br label %111

103:                                              ; preds = %124
  %104 = icmp eq i32 %10, 0
  %105 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %106 = mul nuw nsw i64 %87, 20480
  %107 = zext i32 %76 to i64
  %108 = add nuw nsw i64 %106, %107
  %109 = mul nuw nsw i64 %89, 10240
  %110 = add nuw nsw i64 %109, %107
  br label %156

111:                                              ; preds = %124, %86
  %112 = phi i32 [ 0, %86 ], [ %125, %124 ]
  %113 = add i32 %112, %98
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds bfloat, bfloat addrspace(1)* %91, i64 %114
  %116 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %115, float* noundef nonnull %99) #12
  %117 = getelementptr inbounds bfloat, bfloat addrspace(1)* %93, i64 %114
  %118 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %117, float* noundef nonnull %100) #12
  %119 = lshr i32 %113, 6
  %120 = mul nuw nsw i64 %114, 5
  %121 = lshr exact i64 %120, 3
  %122 = zext i32 %119 to i64
  %123 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %121
  br label %127

124:                                              ; preds = %127
  %125 = add i32 %112, 512
  %126 = icmp ult i32 %125, 2560
  br i1 %126, label %111, label %103, !llvm.loop !62

127:                                              ; preds = %127, %111
  %128 = phi i32 [ 0, %111 ], [ %153, %127 ]
  %129 = add nuw nsw i32 %128, %76
  %130 = zext i32 %129 to i64
  %131 = mul i64 %51, %130
  %132 = getelementptr inbounds i8, i8 addrspace(1)* %123, i64 %131
  %133 = mul i64 %55, %130
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
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %101) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %102) #13
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %132, float* noundef nonnull %99, float* noundef nonnull %100, float noundef %140, float noundef %143, float noundef %116, float noundef %118, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #12
  %144 = load float, float* %16, align 4, !tbaa !58
  %145 = zext i32 %128 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !58
  %148 = fadd float %144, %147
  store float %148, float* %146, align 4, !tbaa !58
  %149 = load float, float* %17, align 4, !tbaa !58
  %150 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %145
  %151 = load float, float* %150, align 4, !tbaa !58
  %152 = fadd float %149, %151
  store float %152, float* %150, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %102) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %101) #13
  %153 = add nuw nsw i32 %128, 1
  %154 = icmp eq i32 %153, 4
  br i1 %154, label %124, label %127, !llvm.loop !63

155:                                              ; preds = %198
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %97) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %95) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %94) #13
  br label %201

156:                                              ; preds = %198, %103
  %157 = phi i16 [ 0, %103 ], [ %199, %198 ]
  %158 = zext i16 %157 to i64
  %159 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %158
  %160 = load float, float* %159, align 4, !tbaa !58
  %161 = call fast float @air.simd_sum.f32(float %160) #14
  br i1 %104, label %162, label %178

162:                                              ; preds = %156
  %163 = fptrunc float %161 to bfloat
  %164 = bitcast float %161 to i32
  %165 = and i32 %164, 2139095040
  %166 = icmp eq i32 %165, 2139095040
  br i1 %166, label %172, label %167

167:                                              ; preds = %162
  %168 = fpext bfloat %163 to float
  %169 = bitcast float %168 to i32
  %170 = and i32 %169, 2139095040
  %171 = icmp eq i32 %170, 2139095040
  br i1 %171, label %172, label %174

172:                                              ; preds = %167, %162
  %173 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %105, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %174

174:                                              ; preds = %172, %167
  %175 = add nuw nsw i64 %108, %158
  %176 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %175
  store bfloat %163, bfloat addrspace(1)* %176, align 2, !tbaa !56
  %177 = getelementptr inbounds float, float addrspace(1)* %6, i64 %175
  store float %161, float addrspace(1)* %177, align 4, !tbaa !58
  br label %178

178:                                              ; preds = %174, %156
  %179 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %158
  %180 = load float, float* %179, align 4, !tbaa !58
  %181 = call fast float @air.simd_sum.f32(float %180) #14
  br i1 %104, label %182, label %198

182:                                              ; preds = %178
  %183 = fptrunc float %181 to bfloat
  %184 = bitcast float %181 to i32
  %185 = and i32 %184, 2139095040
  %186 = icmp eq i32 %185, 2139095040
  br i1 %186, label %192, label %187

187:                                              ; preds = %182
  %188 = fpext bfloat %183 to float
  %189 = bitcast float %188 to i32
  %190 = and i32 %189, 2139095040
  %191 = icmp eq i32 %190, 2139095040
  br i1 %191, label %192, label %194

192:                                              ; preds = %187, %182
  %193 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %105, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %194

194:                                              ; preds = %192, %187
  %195 = add nuw nsw i64 %110, %158
  %196 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %195
  store bfloat %183, bfloat addrspace(1)* %196, align 2, !tbaa !56
  %197 = getelementptr inbounds float, float addrspace(1)* %6, i64 %195
  store float %181, float addrspace(1)* %197, align 4, !tbaa !58
  br label %198

198:                                              ; preds = %194, %178
  %199 = add nuw nsw i16 %157, 1
  %200 = icmp eq i16 %199, 4
  br i1 %200, label %155, label %156, !llvm.loop !64

201:                                              ; preds = %155, %72, %69, %67
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_candidate_probe_q5_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #9
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
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #11
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
  %26 = icmp eq i32 %25, 4
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %36, label %84

28:                                               ; preds = %11
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 3
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !44
  %31 = icmp eq i32 %30, 2560
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %7, i64 0, i32 0
  %33 = load i32, i32 addrspace(2)* %32, align 8
  %34 = icmp eq i32 %33, 4
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
  %83 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt5ELt128EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #12
  br i1 %83, label %89, label %84

84:                                               ; preds = %82, %77, %67, %60, %53, %49, %45, %41, %36, %28, %20, %11
  %85 = icmp eq i32 %10, 0
  br i1 %85, label %86, label %220

86:                                               ; preds = %84
  %87 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %88 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %87, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %220

89:                                               ; preds = %82
  %90 = extractelement <3 x i32> %8, i64 0
  %91 = shl i32 %90, 3
  %92 = shl i32 %9, 2
  %93 = add i32 %91, %92
  %94 = lshr exact i32 %37, 3
  %95 = icmp uge i32 %90, %94
  %96 = extractelement <3 x i32> %8, i64 1
  %97 = icmp ugt i32 %96, 1
  %98 = or i1 %97, %95
  %99 = extractelement <3 x i32> %8, i64 2
  %100 = icmp ne i32 %99, 0
  %101 = or i1 %100, %98
  %102 = icmp uge i32 %93, %37
  %103 = or i1 %101, %102
  br i1 %103, label %220, label %104

104:                                              ; preds = %89
  %105 = zext i32 %96 to i64
  %106 = shl nuw nsw i64 %105, 1
  %107 = or i64 %106, 1
  %108 = mul nuw nsw i64 %106, %63
  %109 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %108
  %110 = mul i64 %107, %63
  %111 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %110
  %112 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %112) #13
  %113 = bitcast [16 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %113) #13
  %114 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %114) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %114, i8 0, i64 16, i1 false)
  %115 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %115) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %115, i8 0, i64 16, i1 false)
  %116 = shl i32 %10, 4
  %117 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %118 = getelementptr inbounds [16 x float], [16 x float]* %13, i64 0, i64 0
  %119 = bitcast float* %16 to i8*
  %120 = bitcast float* %17 to i8*
  br label %130

121:                                              ; preds = %143
  %122 = icmp eq i32 %10, 0
  %123 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %124 = zext i32 %37 to i64
  %125 = mul nuw nsw i64 %106, %124
  %126 = zext i32 %93 to i64
  %127 = add nuw nsw i64 %125, %126
  %128 = mul nuw nsw i64 %107, %124
  %129 = add nuw nsw i64 %128, %126
  br label %175

130:                                              ; preds = %143, %104
  %131 = phi i32 [ 0, %104 ], [ %144, %143 ]
  %132 = add i32 %131, %116
  %133 = zext i32 %132 to i64
  %134 = getelementptr inbounds bfloat, bfloat addrspace(1)* %109, i64 %133
  %135 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %134, float* noundef nonnull %117) #12
  %136 = getelementptr inbounds bfloat, bfloat addrspace(1)* %111, i64 %133
  %137 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %136, float* noundef nonnull %118) #12
  %138 = lshr i32 %132, 7
  %139 = mul nuw nsw i64 %133, 5
  %140 = lshr exact i64 %139, 3
  %141 = zext i32 %138 to i64
  %142 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %140
  br label %146

143:                                              ; preds = %146
  %144 = add i32 %131, 512
  %145 = icmp ult i32 %144, %19
  br i1 %145, label %130, label %121, !llvm.loop !65

146:                                              ; preds = %146, %130
  %147 = phi i32 [ 0, %130 ], [ %172, %146 ]
  %148 = add nuw nsw i32 %147, %93
  %149 = zext i32 %148 to i64
  %150 = mul i64 %62, %149
  %151 = getelementptr inbounds i8, i8 addrspace(1)* %142, i64 %150
  %152 = mul i64 %69, %149
  %153 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %152
  %154 = bitcast i8 addrspace(1)* %153 to bfloat addrspace(1)*
  %155 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %152
  %156 = bitcast i8 addrspace(1)* %155 to bfloat addrspace(1)*
  %157 = getelementptr inbounds bfloat, bfloat addrspace(1)* %154, i64 %141
  %158 = load bfloat, bfloat addrspace(1)* %157, align 2, !tbaa !56
  %159 = fpext bfloat %158 to float
  %160 = getelementptr inbounds bfloat, bfloat addrspace(1)* %156, i64 %141
  %161 = load bfloat, bfloat addrspace(1)* %160, align 2, !tbaa !56
  %162 = fpext bfloat %161 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %119) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %120) #13
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %151, float* noundef nonnull %117, float* noundef nonnull %118, float noundef %159, float noundef %162, float noundef %135, float noundef %137, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #12
  %163 = load float, float* %16, align 4, !tbaa !58
  %164 = zext i32 %147 to i64
  %165 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %164
  %166 = load float, float* %165, align 4, !tbaa !58
  %167 = fadd float %163, %166
  store float %167, float* %165, align 4, !tbaa !58
  %168 = load float, float* %17, align 4, !tbaa !58
  %169 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %164
  %170 = load float, float* %169, align 4, !tbaa !58
  %171 = fadd float %168, %170
  store float %171, float* %169, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %120) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %119) #13
  %172 = add nuw nsw i32 %147, 1
  %173 = icmp eq i32 %172, 4
  br i1 %173, label %143, label %146, !llvm.loop !66

174:                                              ; preds = %217
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %115) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %114) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %113) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %112) #13
  br label %220

175:                                              ; preds = %217, %121
  %176 = phi i16 [ 0, %121 ], [ %218, %217 ]
  %177 = zext i16 %176 to i64
  %178 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %177
  %179 = load float, float* %178, align 4, !tbaa !58
  %180 = call fast float @air.simd_sum.f32(float %179) #14
  br i1 %122, label %181, label %197

181:                                              ; preds = %175
  %182 = fptrunc float %180 to bfloat
  %183 = bitcast float %180 to i32
  %184 = and i32 %183, 2139095040
  %185 = icmp eq i32 %184, 2139095040
  br i1 %185, label %191, label %186

186:                                              ; preds = %181
  %187 = fpext bfloat %182 to float
  %188 = bitcast float %187 to i32
  %189 = and i32 %188, 2139095040
  %190 = icmp eq i32 %189, 2139095040
  br i1 %190, label %191, label %193

191:                                              ; preds = %186, %181
  %192 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %123, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %193

193:                                              ; preds = %191, %186
  %194 = add nuw nsw i64 %127, %177
  %195 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %194
  store bfloat %182, bfloat addrspace(1)* %195, align 2, !tbaa !56
  %196 = getelementptr inbounds float, float addrspace(1)* %6, i64 %194
  store float %180, float addrspace(1)* %196, align 4, !tbaa !58
  br label %197

197:                                              ; preds = %193, %175
  %198 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %177
  %199 = load float, float* %198, align 4, !tbaa !58
  %200 = call fast float @air.simd_sum.f32(float %199) #14
  br i1 %122, label %201, label %217

201:                                              ; preds = %197
  %202 = fptrunc float %200 to bfloat
  %203 = bitcast float %200 to i32
  %204 = and i32 %203, 2139095040
  %205 = icmp eq i32 %204, 2139095040
  br i1 %205, label %211, label %206

206:                                              ; preds = %201
  %207 = fpext bfloat %202 to float
  %208 = bitcast float %207 to i32
  %209 = and i32 %208, 2139095040
  %210 = icmp eq i32 %209, 2139095040
  br i1 %210, label %211, label %213

211:                                              ; preds = %206, %201
  %212 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %123, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %213

213:                                              ; preds = %211, %206
  %214 = add nuw nsw i64 %129, %177
  %215 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %214
  store bfloat %202, bfloat addrspace(1)* %215, align 2, !tbaa !56
  %216 = getelementptr inbounds float, float addrspace(1)* %6, i64 %214
  store float %200, float addrspace(1)* %216, align 4, !tbaa !58
  br label %217

217:                                              ; preds = %213, %197
  %218 = add nuw nsw i16 %176, 1
  %219 = icmp eq i16 %218, 4
  br i1 %219, label %174, label %175, !llvm.loop !67

220:                                              ; preds = %174, %89, %86, %84
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_candidate_probe_q6_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, float addrspace(1)* noundef "air-buffer-no-alias" %8, <3 x i32> noundef %9, <3 x i32> noundef %10, i32 noundef %11, i32 noundef %12, i32 noundef %13) local_unnamed_addr #0 {
  %15 = icmp ne <3 x i32> %10, <i32 64, i32 1, i32 1>
  %16 = tail call i1 @air.any.v3i1(<3 x i1> %15) #9
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
  %27 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %26, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %29

28:                                               ; preds = %14
  tail call void @_ZN28r5_raw_odd_rowpair_tap_sep2212project_pairILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEEPU9MTLdevicefRU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, float addrspace(1)* noundef %8, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %9, i32 noundef %12, i32 noundef %13) #11
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
  %27 = icmp eq i32 %26, 4
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
  %66 = tail call zeroext i1 @_ZN28r5_raw_odd_rowpair_tap_sep2217row_stride_boundsILt6ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7) #12
  br i1 %66, label %72, label %67

67:                                               ; preds = %65, %60, %53, %49, %45, %41, %37, %33, %29, %11
  %68 = icmp eq i32 %10, 0
  br i1 %68, label %69, label %201

69:                                               ; preds = %67
  %70 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %71 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %70, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %201

72:                                               ; preds = %65
  %73 = extractelement <3 x i32> %8, i64 0
  %74 = shl i32 %73, 3
  %75 = shl i32 %9, 2
  %76 = add i32 %74, %75
  %77 = icmp ugt i32 %73, 767
  %78 = extractelement <3 x i32> %8, i64 1
  %79 = icmp ugt i32 %78, 1
  %80 = or i1 %79, %77
  %81 = extractelement <3 x i32> %8, i64 2
  %82 = icmp ne i32 %81, 0
  %83 = or i1 %82, %80
  %84 = icmp ugt i32 %76, 6143
  %85 = or i1 %83, %84
  br i1 %85, label %201, label %86

86:                                               ; preds = %72
  %87 = zext i32 %78 to i64
  %88 = shl nuw nsw i64 %87, 1
  %89 = or i64 %88, 1
  %90 = mul nuw nsw i64 %87, 5120
  %91 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %90
  %92 = mul nuw nsw i64 %89, 2560
  %93 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %92
  %94 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %94) #13
  %95 = bitcast [8 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %95) #13
  %96 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %96, i8 0, i64 16, i1 false)
  %97 = bitcast [4 x float]* %15 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %97) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %97, i8 0, i64 16, i1 false)
  %98 = shl i32 %10, 3
  %99 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %100 = getelementptr inbounds [8 x float], [8 x float]* %13, i64 0, i64 0
  %101 = bitcast float* %16 to i8*
  %102 = bitcast float* %17 to i8*
  br label %111

103:                                              ; preds = %124
  %104 = icmp eq i32 %10, 0
  %105 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %106 = mul nuw nsw i64 %87, 12288
  %107 = zext i32 %76 to i64
  %108 = add nuw nsw i64 %106, %107
  %109 = mul nuw nsw i64 %89, 6144
  %110 = add nuw nsw i64 %109, %107
  br label %156

111:                                              ; preds = %124, %86
  %112 = phi i32 [ 0, %86 ], [ %125, %124 ]
  %113 = add i32 %112, %98
  %114 = zext i32 %113 to i64
  %115 = getelementptr inbounds bfloat, bfloat addrspace(1)* %91, i64 %114
  %116 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %115, float* noundef nonnull %99) #12
  %117 = getelementptr inbounds bfloat, bfloat addrspace(1)* %93, i64 %114
  %118 = call fast float @_ZN22r5_raw_odd_control_tap30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %117, float* noundef nonnull %100) #12
  %119 = lshr i32 %113, 6
  %120 = mul nuw nsw i64 %114, 6
  %121 = lshr exact i64 %120, 3
  %122 = zext i32 %119 to i64
  %123 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %121
  br label %127

124:                                              ; preds = %127
  %125 = add i32 %112, 256
  %126 = icmp ult i32 %125, 2560
  br i1 %126, label %111, label %103, !llvm.loop !68

127:                                              ; preds = %127, %111
  %128 = phi i32 [ 0, %111 ], [ %153, %127 ]
  %129 = add nuw nsw i32 %128, %76
  %130 = zext i32 %129 to i64
  %131 = mul i64 %51, %130
  %132 = getelementptr inbounds i8, i8 addrspace(1)* %123, i64 %131
  %133 = mul i64 %55, %130
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
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %101) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %102) #13
  call void @_ZN28r5_raw_odd_rowpair_tap_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %132, float* noundef nonnull %99, float* noundef nonnull %100, float noundef %140, float noundef %143, float noundef %116, float noundef %118, float* noundef nonnull align 4 dereferenceable(4) %16, float* noundef nonnull align 4 dereferenceable(4) %17) #12
  %144 = load float, float* %16, align 4, !tbaa !58
  %145 = zext i32 %128 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !58
  %148 = fadd float %144, %147
  store float %148, float* %146, align 4, !tbaa !58
  %149 = load float, float* %17, align 4, !tbaa !58
  %150 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %145
  %151 = load float, float* %150, align 4, !tbaa !58
  %152 = fadd float %149, %151
  store float %152, float* %150, align 4, !tbaa !58
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %102) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %101) #13
  %153 = add nuw nsw i32 %128, 1
  %154 = icmp eq i32 %153, 4
  br i1 %154, label %124, label %127, !llvm.loop !69

155:                                              ; preds = %198
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %97) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %95) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %94) #13
  br label %201

156:                                              ; preds = %198, %103
  %157 = phi i16 [ 0, %103 ], [ %199, %198 ]
  %158 = zext i16 %157 to i64
  %159 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %158
  %160 = load float, float* %159, align 4, !tbaa !58
  %161 = call fast float @air.simd_sum.f32(float %160) #14
  br i1 %104, label %162, label %178

162:                                              ; preds = %156
  %163 = fptrunc float %161 to bfloat
  %164 = bitcast float %161 to i32
  %165 = and i32 %164, 2139095040
  %166 = icmp eq i32 %165, 2139095040
  br i1 %166, label %172, label %167

167:                                              ; preds = %162
  %168 = fpext bfloat %163 to float
  %169 = bitcast float %168 to i32
  %170 = and i32 %169, 2139095040
  %171 = icmp eq i32 %170, 2139095040
  br i1 %171, label %172, label %174

172:                                              ; preds = %167, %162
  %173 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %105, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %174

174:                                              ; preds = %172, %167
  %175 = add nuw nsw i64 %108, %158
  %176 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %175
  store bfloat %163, bfloat addrspace(1)* %176, align 2, !tbaa !56
  %177 = getelementptr inbounds float, float addrspace(1)* %6, i64 %175
  store float %161, float addrspace(1)* %177, align 4, !tbaa !58
  br label %178

178:                                              ; preds = %174, %156
  %179 = getelementptr inbounds [4 x float], [4 x float]* %15, i64 0, i64 %158
  %180 = load float, float* %179, align 4, !tbaa !58
  %181 = call fast float @air.simd_sum.f32(float %180) #14
  br i1 %104, label %182, label %198

182:                                              ; preds = %178
  %183 = fptrunc float %181 to bfloat
  %184 = bitcast float %181 to i32
  %185 = and i32 %184, 2139095040
  %186 = icmp eq i32 %185, 2139095040
  br i1 %186, label %192, label %187

187:                                              ; preds = %182
  %188 = fpext bfloat %183 to float
  %189 = bitcast float %188 to i32
  %190 = and i32 %189, 2139095040
  %191 = icmp eq i32 %190, 2139095040
  br i1 %191, label %192, label %194

192:                                              ; preds = %187, %182
  %193 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %105, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %194

194:                                              ; preds = %192, %187
  %195 = add nuw nsw i64 %110, %158
  %196 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %195
  store bfloat %183, bfloat addrspace(1)* %196, align 2, !tbaa !56
  %197 = getelementptr inbounds float, float addrspace(1)* %6, i64 %195
  store float %181, float addrspace(1)* %197, align 4, !tbaa !58
  br label %198

198:                                              ; preds = %194, %178
  %199 = add nuw nsw i16 %157, 1
  %200 = icmp eq i16 %199, 4
  br i1 %200, label %155, label %156, !llvm.loop !70

201:                                              ; preds = %155, %72, %69, %67
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

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #6

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
  %22 = load i8, i8 addrspace(1)* %21, align 1, !tbaa !72
  %23 = or i32 %19, 1
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %24
  %26 = load i8, i8 addrspace(1)* %25, align 1, !tbaa !72
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
  %38 = tail call float @air.convert.f.f32.s.i32(i32 %37) #9
  %39 = or i32 %33, 1
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds float, float* %1, i64 %40
  %42 = load float, float* %41, align 4, !tbaa !58
  %43 = zext i8 %30 to i32
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %43) #9
  %45 = fmul float %42, %44
  %46 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %45)
  %47 = or i32 %33, 2
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds float, float* %1, i64 %48
  %50 = load float, float* %49, align 4, !tbaa !58
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %31) #9
  %52 = tail call float @llvm.fmuladd.f32(float %50, float %51, float %46)
  %53 = or i32 %33, 3
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds float, float* %1, i64 %54
  %56 = load float, float* %55, align 4, !tbaa !58
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %32) #9
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
  br i1 %74, label %10, label %15, !llvm.loop !73
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #7

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #8

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
  br i1 %5, label %4, label %3, !llvm.loop !74
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
  %30 = load i8, i8 addrspace(1)* %29, align 1, !tbaa !72
  %31 = zext i8 %30 to i32
  %32 = and i32 %31, 31
  %33 = and i32 %31, 224
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 1
  %35 = load i8, i8 addrspace(1)* %34, align 1, !tbaa !72
  %36 = zext i8 %35 to i32
  %37 = and i32 %36, 3
  %38 = and i32 %36, 124
  %39 = and i32 %36, 128
  %40 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 2
  %41 = load i8, i8 addrspace(1)* %40, align 1, !tbaa !72
  %42 = zext i8 %41 to i32
  %43 = and i32 %42, 15
  %44 = and i32 %42, 240
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 3
  %46 = load i8, i8 addrspace(1)* %45, align 1, !tbaa !72
  %47 = zext i8 %46 to i32
  %48 = and i32 %47, 1
  %49 = and i32 %47, 62
  %50 = and i32 %47, 192
  %51 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 4
  %52 = load i8, i8 addrspace(1)* %51, align 1, !tbaa !72
  %53 = zext i8 %52 to i32
  %54 = and i32 %53, 7
  %55 = and i32 %53, 248
  %56 = tail call float @air.convert.f.f32.s.i32(i32 %32) #9
  %57 = load float, float* %25, align 4, !tbaa !58
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %19)
  %59 = tail call float @air.convert.f.f32.s.i32(i32 %33) #9
  %60 = getelementptr inbounds float, float* %25, i64 1
  %61 = load float, float* %60, align 4, !tbaa !58
  %62 = tail call float @llvm.fmuladd.f32(float %59, float %61, float %58)
  %63 = tail call float @air.convert.f.f32.s.i32(i32 %37) #9
  %64 = fmul float %61, 2.560000e+02
  %65 = tail call float @llvm.fmuladd.f32(float %63, float %64, float %62)
  %66 = tail call float @air.convert.f.f32.s.i32(i32 %38) #9
  %67 = getelementptr inbounds float, float* %25, i64 2
  %68 = load float, float* %67, align 4, !tbaa !58
  %69 = tail call float @llvm.fmuladd.f32(float %66, float %68, float %65)
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %39) #9
  %71 = getelementptr inbounds float, float* %25, i64 3
  %72 = load float, float* %71, align 4, !tbaa !58
  %73 = tail call float @llvm.fmuladd.f32(float %70, float %72, float %69)
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %43) #9
  %75 = fmul float %72, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %74, float %75, float %73)
  %77 = tail call float @air.convert.f.f32.s.i32(i32 %44) #9
  %78 = getelementptr inbounds float, float* %25, i64 4
  %79 = load float, float* %78, align 4, !tbaa !58
  %80 = tail call float @llvm.fmuladd.f32(float %77, float %79, float %76)
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %48) #9
  %82 = fmul float %79, 2.560000e+02
  %83 = tail call float @llvm.fmuladd.f32(float %81, float %82, float %80)
  %84 = tail call float @air.convert.f.f32.s.i32(i32 %49) #9
  %85 = getelementptr inbounds float, float* %25, i64 5
  %86 = load float, float* %85, align 4, !tbaa !58
  %87 = tail call float @llvm.fmuladd.f32(float %84, float %86, float %83)
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %50) #9
  %89 = getelementptr inbounds float, float* %25, i64 6
  %90 = load float, float* %89, align 4, !tbaa !58
  %91 = tail call float @llvm.fmuladd.f32(float %88, float %90, float %87)
  %92 = tail call float @air.convert.f.f32.s.i32(i32 %54) #9
  %93 = fmul float %90, 2.560000e+02
  %94 = tail call float @llvm.fmuladd.f32(float %92, float %93, float %91)
  %95 = tail call float @air.convert.f.f32.s.i32(i32 %55) #9
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
  br i1 %21, label %15, label %10, !llvm.loop !75
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
  br i1 %5, label %4, label %3, !llvm.loop !76
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
  %30 = load i8, i8 addrspace(1)* %29, align 1, !tbaa !72
  %31 = zext i8 %30 to i32
  %32 = and i32 %31, 63
  %33 = and i32 %31, 192
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 1
  %35 = load i8, i8 addrspace(1)* %34, align 1, !tbaa !72
  %36 = zext i8 %35 to i32
  %37 = and i32 %36, 15
  %38 = and i32 %36, 240
  %39 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 2
  %40 = load i8, i8 addrspace(1)* %39, align 1, !tbaa !72
  %41 = zext i8 %40 to i32
  %42 = and i32 %41, 3
  %43 = and i32 %41, 252
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %32) #9
  %45 = load float, float* %25, align 4, !tbaa !58
  %46 = tail call float @llvm.fmuladd.f32(float %44, float %45, float %19)
  %47 = tail call float @air.convert.f.f32.s.i32(i32 %33) #9
  %48 = getelementptr inbounds float, float* %25, i64 1
  %49 = load float, float* %48, align 4, !tbaa !58
  %50 = tail call float @llvm.fmuladd.f32(float %47, float %49, float %46)
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %37) #9
  %52 = fmul float %49, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %50)
  %54 = tail call float @air.convert.f.f32.s.i32(i32 %38) #9
  %55 = getelementptr inbounds float, float* %25, i64 2
  %56 = load float, float* %55, align 4, !tbaa !58
  %57 = tail call float @llvm.fmuladd.f32(float %54, float %56, float %53)
  %58 = tail call float @air.convert.f.f32.s.i32(i32 %42) #9
  %59 = fmul float %56, 2.560000e+02
  %60 = tail call float @llvm.fmuladd.f32(float %58, float %59, float %57)
  %61 = tail call float @air.convert.f.f32.s.i32(i32 %43) #9
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
  br i1 %21, label %15, label %10, !llvm.loop !77
}

attributes #0 = { convergent mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #1 = { convergent inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="96" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #2 = { mustprogress nofree nosync nounwind readnone willreturn }
attributes #3 = { mustprogress nounwind willreturn }
attributes #4 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #5 = { inlinehint mustprogress nounwind "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" }
attributes #6 = { argmemonly nofree nounwind willreturn writeonly }
attributes #7 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #8 = { convergent mustprogress nounwind willreturn }
attributes #9 = { nounwind readnone willreturn }
attributes #10 = { nounwind willreturn }
attributes #11 = { convergent nobuiltin "no-builtins" }
attributes #12 = { nobuiltin "no-builtins" }
attributes #13 = { nounwind }
attributes #14 = { convergent nounwind willreturn }

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
!9 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_candidate_probe_q4_g64, !10, !11}
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
!28 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_candidate_probe_q5_g64, !10, !11}
!29 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_candidate_probe_q5_g128, !10, !11}
!30 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, float addrspace(1)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_candidate_probe_q6_g64, !10, !11}
!31 = !{!"air.compile.denorms_disable"}
!32 = !{!"air.compile.fast_math_enable"}
!33 = !{!"air.compile.framebuffer_fetch_enable"}
!34 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!35 = !{i32 2, i32 9, i32 0}
!36 = !{!"Metal", i32 4, i32 1, i32 0}
!37 = !{!"/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/candidate_probe.metal"}
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
!72 = !{!41, !41, i64 0}
!73 = distinct !{!73, !55}
!74 = distinct !{!74, !55}
!75 = distinct !{!75, !55}
!76 = distinct !{!76, !55}
!77 = distinct !{!77, !55}
