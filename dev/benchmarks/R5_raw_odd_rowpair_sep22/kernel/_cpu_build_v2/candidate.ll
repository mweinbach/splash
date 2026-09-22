; ModuleID = '/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_odd_rowpair_sep22/kernel/_cpu_build_v2/candidate.air'
source_filename = "/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_odd_rowpair_sep22/kernel/candidate.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

%"struct.metal::_atomic" = type { i32 }
%struct.FlashAffineParams = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64 }

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_timed_q4_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #0 {
  %13 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %14 = tail call i1 @air.any.v3i1(<3 x i1> %13) #10
  br i1 %14, label %15, label %20

15:                                               ; preds = %12
  %16 = icmp eq i32 %11, 0
  br i1 %16, label %17, label %21

17:                                               ; preds = %15
  %18 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %19 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %18, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %21

20:                                               ; preds = %12
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %10, i32 noundef %11) #12
  br label %21

21:                                               ; preds = %20, %17, %15
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, <3 x i32> noundef %7, i32 noundef %8, i32 noundef %9) local_unnamed_addr #1 {
  %11 = alloca [16 x float], align 4
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca float, align 4
  %16 = alloca float, align 4
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !36
  switch i32 %18, label %93 [
    i32 2560, label %27
    i32 6144, label %19
  ]

19:                                               ; preds = %10
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !42
  %22 = icmp eq i32 %21, 2560
  %23 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %24 = load i32, i32 addrspace(2)* %23, align 8
  %25 = icmp eq i32 %24, 5
  %26 = select i1 %22, i1 %25, i1 false
  br i1 %26, label %34, label %93

27:                                               ; preds = %10
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %29 = load i32, i32 addrspace(2)* %28, align 4, !tbaa !42
  switch i32 %29, label %93 [
    i32 10240, label %30
    i32 12288, label %30
  ]

30:                                               ; preds = %27, %27
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %32 = load i32, i32 addrspace(2)* %31, align 8, !tbaa !43
  %33 = icmp eq i32 %32, 5
  br i1 %33, label %34, label %93

34:                                               ; preds = %30, %19
  %35 = phi i32 [ 2560, %19 ], [ %29, %30 ]
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %37 = load i32, i32 addrspace(2)* %36, align 4, !tbaa !44
  %38 = icmp eq i32 %37, 1
  br i1 %38, label %39, label %93

39:                                               ; preds = %34
  %40 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %41 = load i32, i32 addrspace(2)* %40, align 8, !tbaa !45
  %42 = icmp eq i32 %41, 1
  br i1 %42, label %43, label %93

43:                                               ; preds = %39
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %45 = load i32, i32 addrspace(2)* %44, align 4, !tbaa !46
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %47, label %93

47:                                               ; preds = %43
  %48 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %49 = load i32, i32 addrspace(2)* %48, align 4, !tbaa !47
  %50 = icmp eq i32 %49, 4
  br i1 %50, label %51, label %93

51:                                               ; preds = %47
  %52 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %53 = load i32, i32 addrspace(2)* %52, align 8, !tbaa !48
  %54 = icmp eq i32 %53, 64
  %55 = and i32 %18, 511
  %56 = icmp eq i32 %55, 0
  %57 = select i1 %54, i1 %56, i1 false
  %58 = and i32 %35, 7
  %59 = icmp eq i32 %58, 0
  %60 = select i1 %57, i1 %59, i1 false
  br i1 %60, label %61, label %93

61:                                               ; preds = %51
  %62 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %63 = load i64, i64 addrspace(2)* %62, align 8, !tbaa !49
  %64 = zext i32 %18 to i64
  %65 = shl nuw nsw i64 %64, 2
  %66 = lshr i64 %64, 1
  %67 = icmp ult i64 %63, %66
  br i1 %67, label %93, label %68

68:                                               ; preds = %61
  %69 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %70 = load i64, i64 addrspace(2)* %69, align 8, !tbaa !50
  %71 = lshr i32 %18, 5
  %72 = and i32 %71, 134217726
  %73 = zext i32 %72 to i64
  %74 = icmp uge i64 %70, %73
  %75 = and i64 %70, 1
  %76 = icmp eq i64 %75, 0
  %77 = and i1 %74, %76
  br i1 %77, label %78, label %93

78:                                               ; preds = %68
  %79 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %80 = load i64, i64 addrspace(2)* %79, align 8, !tbaa !51
  %81 = and i64 %80, 1
  %82 = icmp eq i64 %81, 0
  br i1 %82, label %83, label %93

83:                                               ; preds = %78
  %84 = xor i64 %66, -1
  %85 = add nsw i32 %35, -1
  %86 = zext i32 %85 to i64
  %87 = udiv i64 %84, %86
  %88 = icmp ugt i64 %63, %87
  br i1 %88, label %93, label %89

89:                                               ; preds = %83
  %90 = xor i64 %73, -1
  %91 = udiv i64 %90, %86
  %92 = icmp ugt i64 %70, %91
  br i1 %92, label %93, label %98

93:                                               ; preds = %89, %83, %78, %68, %61, %51, %47, %43, %39, %34, %30, %27, %19, %10
  %94 = icmp eq i32 %9, 0
  br i1 %94, label %95, label %237

95:                                               ; preds = %93
  %96 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %97 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %96, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %237

98:                                               ; preds = %89
  %99 = extractelement <3 x i32> %7, i64 0
  %100 = shl i32 %99, 3
  %101 = shl i32 %8, 2
  %102 = add i32 %100, %101
  %103 = lshr i32 %35, 3
  %104 = icmp uge i32 %99, %103
  %105 = extractelement <3 x i32> %7, i64 1
  %106 = icmp ugt i32 %105, 2
  %107 = or i1 %106, %104
  %108 = extractelement <3 x i32> %7, i64 2
  %109 = icmp ne i32 %108, 0
  %110 = or i1 %109, %107
  %111 = xor i1 %110, true
  %112 = icmp ult i32 %102, %35
  %113 = select i1 %111, i1 %112, i1 false
  br i1 %113, label %114, label %237

114:                                              ; preds = %98
  %115 = icmp eq i32 %105, 2
  br i1 %115, label %116, label %118

116:                                              ; preds = %114
  %117 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %65
  tail call void @_ZN18r5_raw_odd_literal12project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %117, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %102, i32 noundef 0, i64 noundef 4, i32 noundef %9) #12
  br label %237

118:                                              ; preds = %114
  %119 = zext i32 %105 to i64
  %120 = shl nuw nsw i64 %119, 1
  %121 = or i64 %120, 1
  %122 = mul nuw nsw i64 %120, %64
  %123 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %122
  %124 = mul i64 %121, %64
  %125 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %124
  %126 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %126) #13
  %127 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %127) #13
  %128 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %128) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %128, i8 0, i64 16, i1 false)
  %129 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %129) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %129, i8 0, i64 16, i1 false)
  %130 = shl i32 %9, 4
  %131 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 0
  %132 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %133 = bitcast float* %15 to i8*
  %134 = bitcast float* %16 to i8*
  br label %144

135:                                              ; preds = %156
  %136 = icmp eq i32 %9, 0
  %137 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %138 = zext i32 %35 to i64
  %139 = mul i64 %120, %138
  %140 = zext i32 %102 to i64
  %141 = add i64 %139, %140
  %142 = mul i64 %121, %138
  %143 = add i64 %142, %140
  br label %191

144:                                              ; preds = %156, %118
  %145 = phi i32 [ 0, %118 ], [ %157, %156 ]
  %146 = add i32 %145, %130
  %147 = zext i32 %146 to i64
  %148 = getelementptr inbounds bfloat, bfloat addrspace(1)* %123, i64 %147
  %149 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %148, float* noundef nonnull %131) #14
  %150 = getelementptr inbounds bfloat, bfloat addrspace(1)* %125, i64 %147
  %151 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %150, float* noundef nonnull %132) #14
  %152 = lshr i32 %146, 6
  %153 = lshr exact i64 %147, 1
  %154 = zext i32 %152 to i64
  %155 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %153
  br label %159

156:                                              ; preds = %187
  %157 = add i32 %145, 512
  %158 = icmp ult i32 %157, %18
  br i1 %158, label %144, label %135, !llvm.loop !52

159:                                              ; preds = %187, %144
  %160 = phi i32 [ 0, %144 ], [ %188, %187 ]
  %161 = add nuw nsw i32 %160, %102
  %162 = icmp ult i32 %161, %35
  br i1 %162, label %163, label %187

163:                                              ; preds = %159
  %164 = zext i32 %161 to i64
  %165 = mul i64 %63, %164
  %166 = getelementptr inbounds i8, i8 addrspace(1)* %155, i64 %165
  %167 = mul i64 %70, %164
  %168 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %167
  %169 = bitcast i8 addrspace(1)* %168 to bfloat addrspace(1)*
  %170 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %167
  %171 = bitcast i8 addrspace(1)* %170 to bfloat addrspace(1)*
  %172 = getelementptr inbounds bfloat, bfloat addrspace(1)* %169, i64 %154
  %173 = load bfloat, bfloat addrspace(1)* %172, align 2, !tbaa !54
  %174 = fpext bfloat %173 to float
  %175 = getelementptr inbounds bfloat, bfloat addrspace(1)* %171, i64 %154
  %176 = load bfloat, bfloat addrspace(1)* %175, align 2, !tbaa !54
  %177 = fpext bfloat %176 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %133) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %134) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %166, float* noundef nonnull %131, float* noundef nonnull %132, float noundef %174, float noundef %177, float noundef %149, float noundef %151, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #14
  %178 = load float, float* %15, align 4, !tbaa !56
  %179 = zext i32 %160 to i64
  %180 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %179
  %181 = load float, float* %180, align 4, !tbaa !56
  %182 = fadd float %178, %181
  store float %182, float* %180, align 4, !tbaa !56
  %183 = load float, float* %16, align 4, !tbaa !56
  %184 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %179
  %185 = load float, float* %184, align 4, !tbaa !56
  %186 = fadd float %183, %185
  store float %186, float* %184, align 4, !tbaa !56
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %134) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %133) #13
  br label %187

187:                                              ; preds = %163, %159
  %188 = add nuw nsw i32 %160, 1
  %189 = icmp eq i32 %188, 4
  br i1 %189, label %156, label %159, !llvm.loop !58

190:                                              ; preds = %234
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %129) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %128) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %127) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %126) #13
  br label %237

191:                                              ; preds = %234, %135
  %192 = phi i32 [ 0, %135 ], [ %235, %234 ]
  %193 = add nuw nsw i32 %192, %102
  %194 = icmp ult i32 %193, %35
  br i1 %194, label %195, label %234

195:                                              ; preds = %191
  %196 = zext i32 %192 to i64
  %197 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %196
  %198 = load float, float* %197, align 4, !tbaa !56
  %199 = call fast float @air.simd_sum.f32(float %198) #15
  br i1 %136, label %200, label %215

200:                                              ; preds = %195
  %201 = fptrunc float %199 to bfloat
  %202 = bitcast float %199 to i32
  %203 = and i32 %202, 2139095040
  %204 = icmp eq i32 %203, 2139095040
  br i1 %204, label %210, label %205

205:                                              ; preds = %200
  %206 = fpext bfloat %201 to float
  %207 = bitcast float %206 to i32
  %208 = and i32 %207, 2139095040
  %209 = icmp eq i32 %208, 2139095040
  br i1 %209, label %210, label %212

210:                                              ; preds = %205, %200
  %211 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %137, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %212

212:                                              ; preds = %210, %205
  %213 = add i64 %141, %196
  %214 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %213
  store bfloat %201, bfloat addrspace(1)* %214, align 2, !tbaa !54
  br label %215

215:                                              ; preds = %212, %195
  %216 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %196
  %217 = load float, float* %216, align 4, !tbaa !56
  %218 = call fast float @air.simd_sum.f32(float %217) #15
  br i1 %136, label %219, label %234

219:                                              ; preds = %215
  %220 = fptrunc float %218 to bfloat
  %221 = bitcast float %218 to i32
  %222 = and i32 %221, 2139095040
  %223 = icmp eq i32 %222, 2139095040
  br i1 %223, label %229, label %224

224:                                              ; preds = %219
  %225 = fpext bfloat %220 to float
  %226 = bitcast float %225 to i32
  %227 = and i32 %226, 2139095040
  %228 = icmp eq i32 %227, 2139095040
  br i1 %228, label %229, label %231

229:                                              ; preds = %224, %219
  %230 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %137, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %231

231:                                              ; preds = %229, %224
  %232 = add i64 %143, %196
  %233 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %232
  store bfloat %220, bfloat addrspace(1)* %233, align 2, !tbaa !54
  br label %234

234:                                              ; preds = %231, %215, %191
  %235 = add nuw nsw i32 %192, 1
  %236 = icmp eq i32 %235, 4
  br i1 %236, label %190, label %191, !llvm.loop !59

237:                                              ; preds = %190, %116, %98, %95, %93
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_timed_q5_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #0 {
  %13 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %14 = tail call i1 @air.any.v3i1(<3 x i1> %13) #10
  br i1 %14, label %15, label %20

15:                                               ; preds = %12
  %16 = icmp eq i32 %11, 0
  br i1 %16, label %17, label %21

17:                                               ; preds = %15
  %18 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %19 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %18, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %21

20:                                               ; preds = %12
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %10, i32 noundef %11) #12
  br label %21

21:                                               ; preds = %20, %17, %15
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, <3 x i32> noundef %7, i32 noundef %8, i32 noundef %9) local_unnamed_addr #1 {
  %11 = alloca [16 x float], align 4
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca float, align 4
  %16 = alloca float, align 4
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !36
  %19 = icmp eq i32 %18, 2560
  br i1 %19, label %20, label %68

20:                                               ; preds = %10
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !42
  %23 = icmp eq i32 %22, 10240
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %25 = load i32, i32 addrspace(2)* %24, align 8
  %26 = icmp eq i32 %25, 5
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %28, label %68

28:                                               ; preds = %20
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !44
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %68

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %34 = load i32, i32 addrspace(2)* %33, align 8, !tbaa !45
  %35 = icmp eq i32 %34, 1
  br i1 %35, label %36, label %68

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !46
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %68

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %42 = load i32, i32 addrspace(2)* %41, align 4, !tbaa !47
  %43 = icmp eq i32 %42, 5
  br i1 %43, label %44, label %68

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %46 = load i32, i32 addrspace(2)* %45, align 8, !tbaa !48
  %47 = icmp eq i32 %46, 64
  br i1 %47, label %48, label %68

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %50 = load i64, i64 addrspace(2)* %49, align 8, !tbaa !49
  %51 = icmp ult i64 %50, 1600
  br i1 %51, label %68, label %52

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %54 = load i64, i64 addrspace(2)* %53, align 8, !tbaa !50
  %55 = icmp ugt i64 %54, 79
  %56 = and i64 %54, 1
  %57 = icmp eq i64 %56, 0
  %58 = and i1 %55, %57
  br i1 %58, label %59, label %68

59:                                               ; preds = %52
  %60 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %61 = load i64, i64 addrspace(2)* %60, align 8, !tbaa !51
  %62 = and i64 %61, 1
  %63 = icmp ne i64 %62, 0
  %64 = icmp ugt i64 %50, 1801615789990189
  %65 = select i1 %63, i1 true, i1 %64
  %66 = icmp ugt i64 %54, 1801615789990189
  %67 = select i1 %65, i1 true, i1 %66
  br i1 %67, label %68, label %73

68:                                               ; preds = %59, %52, %48, %44, %40, %36, %32, %28, %20, %10
  %69 = icmp eq i32 %9, 0
  br i1 %69, label %70, label %210

70:                                               ; preds = %68
  %71 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %72 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %71, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %210

73:                                               ; preds = %59
  %74 = extractelement <3 x i32> %7, i64 0
  %75 = shl i32 %74, 3
  %76 = shl i32 %8, 2
  %77 = add i32 %75, %76
  %78 = icmp ugt i32 %74, 1279
  %79 = extractelement <3 x i32> %7, i64 1
  %80 = icmp ugt i32 %79, 2
  %81 = or i1 %80, %78
  %82 = extractelement <3 x i32> %7, i64 2
  %83 = icmp ne i32 %82, 0
  %84 = or i1 %83, %81
  %85 = icmp ugt i32 %77, 10239
  %86 = or i1 %84, %85
  br i1 %86, label %210, label %87

87:                                               ; preds = %73
  %88 = icmp eq i32 %79, 2
  br i1 %88, label %89, label %91

89:                                               ; preds = %87
  %90 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 10240
  tail call void @_ZN18r5_raw_odd_literal12project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %90, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %77, i32 noundef 0, i64 noundef 4, i32 noundef %9) #12
  br label %210

91:                                               ; preds = %87
  %92 = zext i32 %79 to i64
  %93 = shl nuw nsw i64 %92, 1
  %94 = or i64 %93, 1
  %95 = mul nuw nsw i64 %92, 5120
  %96 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %95
  %97 = mul nuw nsw i64 %94, 2560
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %97
  %99 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %99) #13
  %100 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %100) #13
  %101 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %101) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %101, i8 0, i64 16, i1 false)
  %102 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %102) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %102, i8 0, i64 16, i1 false)
  %103 = shl i32 %9, 4
  %104 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 0
  %105 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %106 = bitcast float* %15 to i8*
  %107 = bitcast float* %16 to i8*
  br label %116

108:                                              ; preds = %129
  %109 = icmp eq i32 %9, 0
  %110 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %111 = mul nuw nsw i64 %92, 20480
  %112 = zext i32 %77 to i64
  %113 = add nuw nsw i64 %111, %112
  %114 = mul nuw nsw i64 %94, 10240
  %115 = add nuw nsw i64 %114, %112
  br label %164

116:                                              ; preds = %129, %91
  %117 = phi i32 [ 0, %91 ], [ %130, %129 ]
  %118 = add i32 %117, %103
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds bfloat, bfloat addrspace(1)* %96, i64 %119
  %121 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %120, float* noundef nonnull %104) #14
  %122 = getelementptr inbounds bfloat, bfloat addrspace(1)* %98, i64 %119
  %123 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %122, float* noundef nonnull %105) #14
  %124 = lshr i32 %118, 6
  %125 = mul nuw nsw i64 %119, 5
  %126 = lshr exact i64 %125, 3
  %127 = zext i32 %124 to i64
  %128 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %126
  br label %132

129:                                              ; preds = %160
  %130 = add i32 %117, 512
  %131 = icmp ult i32 %130, 2560
  br i1 %131, label %116, label %108, !llvm.loop !60

132:                                              ; preds = %160, %116
  %133 = phi i32 [ 0, %116 ], [ %161, %160 ]
  %134 = add nuw nsw i32 %133, %77
  %135 = icmp ult i32 %134, 10240
  br i1 %135, label %136, label %160

136:                                              ; preds = %132
  %137 = zext i32 %134 to i64
  %138 = mul i64 %50, %137
  %139 = getelementptr inbounds i8, i8 addrspace(1)* %128, i64 %138
  %140 = mul i64 %54, %137
  %141 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %140
  %142 = bitcast i8 addrspace(1)* %141 to bfloat addrspace(1)*
  %143 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %140
  %144 = bitcast i8 addrspace(1)* %143 to bfloat addrspace(1)*
  %145 = getelementptr inbounds bfloat, bfloat addrspace(1)* %142, i64 %127
  %146 = load bfloat, bfloat addrspace(1)* %145, align 2, !tbaa !54
  %147 = fpext bfloat %146 to float
  %148 = getelementptr inbounds bfloat, bfloat addrspace(1)* %144, i64 %127
  %149 = load bfloat, bfloat addrspace(1)* %148, align 2, !tbaa !54
  %150 = fpext bfloat %149 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %106) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %107) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %139, float* noundef nonnull %104, float* noundef nonnull %105, float noundef %147, float noundef %150, float noundef %121, float noundef %123, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #14
  %151 = load float, float* %15, align 4, !tbaa !56
  %152 = zext i32 %133 to i64
  %153 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %152
  %154 = load float, float* %153, align 4, !tbaa !56
  %155 = fadd float %151, %154
  store float %155, float* %153, align 4, !tbaa !56
  %156 = load float, float* %16, align 4, !tbaa !56
  %157 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %152
  %158 = load float, float* %157, align 4, !tbaa !56
  %159 = fadd float %156, %158
  store float %159, float* %157, align 4, !tbaa !56
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %107) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %106) #13
  br label %160

160:                                              ; preds = %136, %132
  %161 = add nuw nsw i32 %133, 1
  %162 = icmp eq i32 %161, 4
  br i1 %162, label %129, label %132, !llvm.loop !61

163:                                              ; preds = %207
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %102) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %101) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %100) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %99) #13
  br label %210

164:                                              ; preds = %207, %108
  %165 = phi i32 [ 0, %108 ], [ %208, %207 ]
  %166 = add nuw nsw i32 %165, %77
  %167 = icmp ult i32 %166, 10240
  br i1 %167, label %168, label %207

168:                                              ; preds = %164
  %169 = zext i32 %165 to i64
  %170 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %169
  %171 = load float, float* %170, align 4, !tbaa !56
  %172 = call fast float @air.simd_sum.f32(float %171) #15
  br i1 %109, label %173, label %188

173:                                              ; preds = %168
  %174 = fptrunc float %172 to bfloat
  %175 = bitcast float %172 to i32
  %176 = and i32 %175, 2139095040
  %177 = icmp eq i32 %176, 2139095040
  br i1 %177, label %183, label %178

178:                                              ; preds = %173
  %179 = fpext bfloat %174 to float
  %180 = bitcast float %179 to i32
  %181 = and i32 %180, 2139095040
  %182 = icmp eq i32 %181, 2139095040
  br i1 %182, label %183, label %185

183:                                              ; preds = %178, %173
  %184 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %110, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %185

185:                                              ; preds = %183, %178
  %186 = add nuw nsw i64 %113, %169
  %187 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %186
  store bfloat %174, bfloat addrspace(1)* %187, align 2, !tbaa !54
  br label %188

188:                                              ; preds = %185, %168
  %189 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %169
  %190 = load float, float* %189, align 4, !tbaa !56
  %191 = call fast float @air.simd_sum.f32(float %190) #15
  br i1 %109, label %192, label %207

192:                                              ; preds = %188
  %193 = fptrunc float %191 to bfloat
  %194 = bitcast float %191 to i32
  %195 = and i32 %194, 2139095040
  %196 = icmp eq i32 %195, 2139095040
  br i1 %196, label %202, label %197

197:                                              ; preds = %192
  %198 = fpext bfloat %193 to float
  %199 = bitcast float %198 to i32
  %200 = and i32 %199, 2139095040
  %201 = icmp eq i32 %200, 2139095040
  br i1 %201, label %202, label %204

202:                                              ; preds = %197, %192
  %203 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %110, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %204

204:                                              ; preds = %202, %197
  %205 = add nuw nsw i64 %115, %169
  %206 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %205
  store bfloat %193, bfloat addrspace(1)* %206, align 2, !tbaa !54
  br label %207

207:                                              ; preds = %204, %188, %164
  %208 = add nuw nsw i32 %165, 1
  %209 = icmp eq i32 %208, 4
  br i1 %209, label %163, label %164, !llvm.loop !62

210:                                              ; preds = %163, %89, %73, %70, %68
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_timed_q5_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #0 {
  %13 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %14 = tail call i1 @air.any.v3i1(<3 x i1> %13) #10
  br i1 %14, label %15, label %20

15:                                               ; preds = %12
  %16 = icmp eq i32 %11, 0
  br i1 %16, label %17, label %21

17:                                               ; preds = %15
  %18 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %19 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %18, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %21

20:                                               ; preds = %12
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %10, i32 noundef %11) #12
  br label %21

21:                                               ; preds = %20, %17, %15
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, <3 x i32> noundef %7, i32 noundef %8, i32 noundef %9) local_unnamed_addr #1 {
  %11 = alloca [16 x float], align 4
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca float, align 4
  %16 = alloca float, align 4
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !36
  switch i32 %18, label %91 [
    i32 6144, label %27
    i32 2560, label %19
  ]

19:                                               ; preds = %10
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !42
  %22 = icmp eq i32 %21, 6144
  %23 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %24 = load i32, i32 addrspace(2)* %23, align 8
  %25 = icmp eq i32 %24, 5
  %26 = select i1 %22, i1 %25, i1 false
  br i1 %26, label %35, label %91

27:                                               ; preds = %10
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %29 = load i32, i32 addrspace(2)* %28, align 4, !tbaa !42
  %30 = icmp eq i32 %29, 2560
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %32 = load i32, i32 addrspace(2)* %31, align 8
  %33 = icmp eq i32 %32, 5
  %34 = select i1 %30, i1 %33, i1 false
  br i1 %34, label %35, label %91

35:                                               ; preds = %27, %19
  %36 = phi i32 [ 6144, %19 ], [ 2560, %27 ]
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !44
  %39 = icmp eq i32 %38, 1
  br i1 %39, label %40, label %91

40:                                               ; preds = %35
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %42 = load i32, i32 addrspace(2)* %41, align 8, !tbaa !45
  %43 = icmp eq i32 %42, 1
  br i1 %43, label %44, label %91

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %46 = load i32, i32 addrspace(2)* %45, align 4, !tbaa !46
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %48, label %91

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %50 = load i32, i32 addrspace(2)* %49, align 4, !tbaa !47
  %51 = icmp eq i32 %50, 5
  br i1 %51, label %52, label %91

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %54 = load i32, i32 addrspace(2)* %53, align 8, !tbaa !48
  %55 = icmp eq i32 %54, 128
  %56 = and i32 %18, 511
  %57 = icmp eq i32 %56, 0
  %58 = select i1 %55, i1 %57, i1 false
  br i1 %58, label %59, label %91

59:                                               ; preds = %52
  %60 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %61 = load i64, i64 addrspace(2)* %60, align 8, !tbaa !49
  %62 = zext i32 %18 to i64
  %63 = mul nuw nsw i64 %62, 5
  %64 = lshr i64 %63, 3
  %65 = icmp ult i64 %61, %64
  br i1 %65, label %91, label %66

66:                                               ; preds = %59
  %67 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %68 = load i64, i64 addrspace(2)* %67, align 8, !tbaa !50
  %69 = lshr i32 %18, 6
  %70 = and i32 %69, 67108862
  %71 = zext i32 %70 to i64
  %72 = icmp uge i64 %68, %71
  %73 = and i64 %68, 1
  %74 = icmp eq i64 %73, 0
  %75 = and i1 %72, %74
  br i1 %75, label %76, label %91

76:                                               ; preds = %66
  %77 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %78 = load i64, i64 addrspace(2)* %77, align 8, !tbaa !51
  %79 = and i64 %78, 1
  %80 = icmp eq i64 %79, 0
  br i1 %80, label %81, label %91

81:                                               ; preds = %76
  %82 = xor i64 %64, -1
  %83 = add nsw i32 %36, -1
  %84 = zext i32 %83 to i64
  %85 = udiv i64 %82, %84
  %86 = icmp ugt i64 %61, %85
  br i1 %86, label %91, label %87

87:                                               ; preds = %81
  %88 = xor i64 %71, -1
  %89 = udiv i64 %88, %84
  %90 = icmp ugt i64 %68, %89
  br i1 %90, label %91, label %96

91:                                               ; preds = %87, %81, %76, %66, %59, %52, %48, %44, %40, %35, %27, %19, %10
  %92 = icmp eq i32 %9, 0
  br i1 %92, label %93, label %236

93:                                               ; preds = %91
  %94 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %95 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %94, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %236

96:                                               ; preds = %87
  %97 = extractelement <3 x i32> %7, i64 0
  %98 = shl i32 %97, 3
  %99 = shl i32 %8, 2
  %100 = add i32 %98, %99
  %101 = lshr exact i32 %36, 3
  %102 = icmp uge i32 %97, %101
  %103 = extractelement <3 x i32> %7, i64 1
  %104 = icmp ugt i32 %103, 2
  %105 = or i1 %104, %102
  %106 = extractelement <3 x i32> %7, i64 2
  %107 = icmp ne i32 %106, 0
  %108 = or i1 %107, %105
  %109 = icmp uge i32 %100, %36
  %110 = or i1 %108, %109
  br i1 %110, label %236, label %111

111:                                              ; preds = %96
  %112 = icmp eq i32 %103, 2
  br i1 %112, label %113, label %116

113:                                              ; preds = %111
  %114 = shl nuw nsw i64 %62, 2
  %115 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %114
  tail call void @_ZN18r5_raw_odd_literal12project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %115, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %100, i32 noundef 0, i64 noundef 4, i32 noundef %9) #12
  br label %236

116:                                              ; preds = %111
  %117 = zext i32 %103 to i64
  %118 = shl nuw nsw i64 %117, 1
  %119 = or i64 %118, 1
  %120 = mul nuw nsw i64 %118, %62
  %121 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %120
  %122 = mul i64 %119, %62
  %123 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %122
  %124 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %124) #13
  %125 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %125) #13
  %126 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %126) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %126, i8 0, i64 16, i1 false)
  %127 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %127) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %127, i8 0, i64 16, i1 false)
  %128 = shl i32 %9, 4
  %129 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 0
  %130 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %131 = bitcast float* %15 to i8*
  %132 = bitcast float* %16 to i8*
  br label %142

133:                                              ; preds = %155
  %134 = icmp eq i32 %9, 0
  %135 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %136 = zext i32 %36 to i64
  %137 = mul nuw nsw i64 %118, %136
  %138 = zext i32 %100 to i64
  %139 = add nuw nsw i64 %137, %138
  %140 = mul nuw nsw i64 %119, %136
  %141 = add nuw nsw i64 %140, %138
  br label %190

142:                                              ; preds = %155, %116
  %143 = phi i32 [ 0, %116 ], [ %156, %155 ]
  %144 = add i32 %143, %128
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds bfloat, bfloat addrspace(1)* %121, i64 %145
  %147 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %146, float* noundef nonnull %129) #14
  %148 = getelementptr inbounds bfloat, bfloat addrspace(1)* %123, i64 %145
  %149 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %148, float* noundef nonnull %130) #14
  %150 = lshr i32 %144, 7
  %151 = mul nuw nsw i64 %145, 5
  %152 = lshr exact i64 %151, 3
  %153 = zext i32 %150 to i64
  %154 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %152
  br label %158

155:                                              ; preds = %186
  %156 = add i32 %143, 512
  %157 = icmp ult i32 %156, %18
  br i1 %157, label %142, label %133, !llvm.loop !63

158:                                              ; preds = %186, %142
  %159 = phi i32 [ 0, %142 ], [ %187, %186 ]
  %160 = add nuw nsw i32 %159, %100
  %161 = icmp ult i32 %160, %36
  br i1 %161, label %162, label %186

162:                                              ; preds = %158
  %163 = zext i32 %160 to i64
  %164 = mul i64 %61, %163
  %165 = getelementptr inbounds i8, i8 addrspace(1)* %154, i64 %164
  %166 = mul i64 %68, %163
  %167 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %166
  %168 = bitcast i8 addrspace(1)* %167 to bfloat addrspace(1)*
  %169 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %166
  %170 = bitcast i8 addrspace(1)* %169 to bfloat addrspace(1)*
  %171 = getelementptr inbounds bfloat, bfloat addrspace(1)* %168, i64 %153
  %172 = load bfloat, bfloat addrspace(1)* %171, align 2, !tbaa !54
  %173 = fpext bfloat %172 to float
  %174 = getelementptr inbounds bfloat, bfloat addrspace(1)* %170, i64 %153
  %175 = load bfloat, bfloat addrspace(1)* %174, align 2, !tbaa !54
  %176 = fpext bfloat %175 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %131) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %132) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %165, float* noundef nonnull %129, float* noundef nonnull %130, float noundef %173, float noundef %176, float noundef %147, float noundef %149, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #14
  %177 = load float, float* %15, align 4, !tbaa !56
  %178 = zext i32 %159 to i64
  %179 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %178
  %180 = load float, float* %179, align 4, !tbaa !56
  %181 = fadd float %177, %180
  store float %181, float* %179, align 4, !tbaa !56
  %182 = load float, float* %16, align 4, !tbaa !56
  %183 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %178
  %184 = load float, float* %183, align 4, !tbaa !56
  %185 = fadd float %182, %184
  store float %185, float* %183, align 4, !tbaa !56
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %132) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %131) #13
  br label %186

186:                                              ; preds = %162, %158
  %187 = add nuw nsw i32 %159, 1
  %188 = icmp eq i32 %187, 4
  br i1 %188, label %155, label %158, !llvm.loop !64

189:                                              ; preds = %233
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %127) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %126) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %125) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %124) #13
  br label %236

190:                                              ; preds = %233, %133
  %191 = phi i32 [ 0, %133 ], [ %234, %233 ]
  %192 = add nuw nsw i32 %191, %100
  %193 = icmp ult i32 %192, %36
  br i1 %193, label %194, label %233

194:                                              ; preds = %190
  %195 = zext i32 %191 to i64
  %196 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %195
  %197 = load float, float* %196, align 4, !tbaa !56
  %198 = call fast float @air.simd_sum.f32(float %197) #15
  br i1 %134, label %199, label %214

199:                                              ; preds = %194
  %200 = fptrunc float %198 to bfloat
  %201 = bitcast float %198 to i32
  %202 = and i32 %201, 2139095040
  %203 = icmp eq i32 %202, 2139095040
  br i1 %203, label %209, label %204

204:                                              ; preds = %199
  %205 = fpext bfloat %200 to float
  %206 = bitcast float %205 to i32
  %207 = and i32 %206, 2139095040
  %208 = icmp eq i32 %207, 2139095040
  br i1 %208, label %209, label %211

209:                                              ; preds = %204, %199
  %210 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %135, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %211

211:                                              ; preds = %209, %204
  %212 = add nuw nsw i64 %139, %195
  %213 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %212
  store bfloat %200, bfloat addrspace(1)* %213, align 2, !tbaa !54
  br label %214

214:                                              ; preds = %211, %194
  %215 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %195
  %216 = load float, float* %215, align 4, !tbaa !56
  %217 = call fast float @air.simd_sum.f32(float %216) #15
  br i1 %134, label %218, label %233

218:                                              ; preds = %214
  %219 = fptrunc float %217 to bfloat
  %220 = bitcast float %217 to i32
  %221 = and i32 %220, 2139095040
  %222 = icmp eq i32 %221, 2139095040
  br i1 %222, label %228, label %223

223:                                              ; preds = %218
  %224 = fpext bfloat %219 to float
  %225 = bitcast float %224 to i32
  %226 = and i32 %225, 2139095040
  %227 = icmp eq i32 %226, 2139095040
  br i1 %227, label %228, label %230

228:                                              ; preds = %223, %218
  %229 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %135, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %230

230:                                              ; preds = %228, %223
  %231 = add nuw nsw i64 %141, %195
  %232 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %231
  store bfloat %219, bfloat addrspace(1)* %232, align 2, !tbaa !54
  br label %233

233:                                              ; preds = %230, %214, %190
  %234 = add nuw nsw i32 %191, 1
  %235 = icmp eq i32 %234, 4
  br i1 %235, label %189, label %190, !llvm.loop !65

236:                                              ; preds = %189, %113, %96, %93, %91
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @r5_raw_odd_rowpair_sep22_timed_q6_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11) local_unnamed_addr #0 {
  %13 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %14 = tail call i1 @air.any.v3i1(<3 x i1> %13) #10
  br i1 %14, label %15, label %20

15:                                               ; preds = %12
  %16 = icmp eq i32 %11, 0
  br i1 %16, label %17, label %21

17:                                               ; preds = %15
  %18 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %19 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %18, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %21

20:                                               ; preds = %12
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %10, i32 noundef %11) #12
  br label %21

21:                                               ; preds = %20, %17, %15
  ret void
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, <3 x i32> noundef %7, i32 noundef %8, i32 noundef %9) local_unnamed_addr #1 {
  %11 = alloca [8 x float], align 4
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = alloca [4 x float], align 4
  %15 = alloca float, align 4
  %16 = alloca float, align 4
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !36
  %19 = icmp eq i32 %18, 2560
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4
  %22 = icmp eq i32 %21, 6144
  %23 = select i1 %19, i1 %22, i1 false
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %25 = load i32, i32 addrspace(2)* %24, align 8
  %26 = icmp eq i32 %25, 5
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %28, label %68

28:                                               ; preds = %10
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !44
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %68

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %34 = load i32, i32 addrspace(2)* %33, align 8, !tbaa !45
  %35 = icmp eq i32 %34, 1
  br i1 %35, label %36, label %68

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !46
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %68

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %42 = load i32, i32 addrspace(2)* %41, align 4, !tbaa !47
  %43 = icmp eq i32 %42, 6
  br i1 %43, label %44, label %68

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %46 = load i32, i32 addrspace(2)* %45, align 8, !tbaa !48
  %47 = icmp eq i32 %46, 64
  br i1 %47, label %48, label %68

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %50 = load i64, i64 addrspace(2)* %49, align 8, !tbaa !49
  %51 = icmp ult i64 %50, 1920
  br i1 %51, label %68, label %52

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %54 = load i64, i64 addrspace(2)* %53, align 8, !tbaa !50
  %55 = icmp ugt i64 %54, 79
  %56 = and i64 %54, 1
  %57 = icmp eq i64 %56, 0
  %58 = and i1 %55, %57
  br i1 %58, label %59, label %68

59:                                               ; preds = %52
  %60 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %61 = load i64, i64 addrspace(2)* %60, align 8, !tbaa !51
  %62 = and i64 %61, 1
  %63 = icmp ne i64 %62, 0
  %64 = icmp ugt i64 %50, 3002888502964276
  %65 = select i1 %63, i1 true, i1 %64
  %66 = icmp ugt i64 %54, 3002888502964276
  %67 = select i1 %65, i1 true, i1 %66
  br i1 %67, label %68, label %73

68:                                               ; preds = %59, %52, %48, %44, %40, %36, %32, %28, %10
  %69 = icmp eq i32 %9, 0
  br i1 %69, label %70, label %210

70:                                               ; preds = %68
  %71 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %72 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %71, i32 2, i32 0, i32 2, i32 0, i1 false) #11
  br label %210

73:                                               ; preds = %59
  %74 = extractelement <3 x i32> %7, i64 0
  %75 = shl i32 %74, 3
  %76 = shl i32 %8, 2
  %77 = add i32 %75, %76
  %78 = icmp ugt i32 %74, 767
  %79 = extractelement <3 x i32> %7, i64 1
  %80 = icmp ugt i32 %79, 2
  %81 = or i1 %80, %78
  %82 = extractelement <3 x i32> %7, i64 2
  %83 = icmp ne i32 %82, 0
  %84 = or i1 %83, %81
  %85 = icmp ugt i32 %77, 6143
  %86 = or i1 %84, %85
  br i1 %86, label %210, label %87

87:                                               ; preds = %73
  %88 = icmp eq i32 %79, 2
  br i1 %88, label %89, label %91

89:                                               ; preds = %87
  %90 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 10240
  tail call void @_ZN18r5_raw_odd_literal12project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %90, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %77, i32 noundef 0, i64 noundef 4, i32 noundef %9) #12
  br label %210

91:                                               ; preds = %87
  %92 = zext i32 %79 to i64
  %93 = shl nuw nsw i64 %92, 1
  %94 = or i64 %93, 1
  %95 = mul nuw nsw i64 %92, 5120
  %96 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %95
  %97 = mul nuw nsw i64 %94, 2560
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %97
  %99 = bitcast [8 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %99) #13
  %100 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %100) #13
  %101 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %101) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %101, i8 0, i64 16, i1 false)
  %102 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %102) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %102, i8 0, i64 16, i1 false)
  %103 = shl i32 %9, 3
  %104 = getelementptr inbounds [8 x float], [8 x float]* %11, i64 0, i64 0
  %105 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %106 = bitcast float* %15 to i8*
  %107 = bitcast float* %16 to i8*
  br label %116

108:                                              ; preds = %129
  %109 = icmp eq i32 %9, 0
  %110 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %111 = mul nuw nsw i64 %92, 12288
  %112 = zext i32 %77 to i64
  %113 = add nuw nsw i64 %111, %112
  %114 = mul nuw nsw i64 %94, 6144
  %115 = add nuw nsw i64 %114, %112
  br label %164

116:                                              ; preds = %129, %91
  %117 = phi i32 [ 0, %91 ], [ %130, %129 ]
  %118 = add i32 %117, %103
  %119 = zext i32 %118 to i64
  %120 = getelementptr inbounds bfloat, bfloat addrspace(1)* %96, i64 %119
  %121 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %120, float* noundef nonnull %104) #14
  %122 = getelementptr inbounds bfloat, bfloat addrspace(1)* %98, i64 %119
  %123 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %122, float* noundef nonnull %105) #14
  %124 = lshr i32 %118, 6
  %125 = mul nuw nsw i64 %119, 6
  %126 = lshr exact i64 %125, 3
  %127 = zext i32 %124 to i64
  %128 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %126
  br label %132

129:                                              ; preds = %160
  %130 = add i32 %117, 256
  %131 = icmp ult i32 %130, 2560
  br i1 %131, label %116, label %108, !llvm.loop !66

132:                                              ; preds = %160, %116
  %133 = phi i32 [ 0, %116 ], [ %161, %160 ]
  %134 = add nuw nsw i32 %133, %77
  %135 = icmp ult i32 %134, 6144
  br i1 %135, label %136, label %160

136:                                              ; preds = %132
  %137 = zext i32 %134 to i64
  %138 = mul i64 %50, %137
  %139 = getelementptr inbounds i8, i8 addrspace(1)* %128, i64 %138
  %140 = mul i64 %54, %137
  %141 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %140
  %142 = bitcast i8 addrspace(1)* %141 to bfloat addrspace(1)*
  %143 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %140
  %144 = bitcast i8 addrspace(1)* %143 to bfloat addrspace(1)*
  %145 = getelementptr inbounds bfloat, bfloat addrspace(1)* %142, i64 %127
  %146 = load bfloat, bfloat addrspace(1)* %145, align 2, !tbaa !54
  %147 = fpext bfloat %146 to float
  %148 = getelementptr inbounds bfloat, bfloat addrspace(1)* %144, i64 %127
  %149 = load bfloat, bfloat addrspace(1)* %148, align 2, !tbaa !54
  %150 = fpext bfloat %149 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %106) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %107) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %139, float* noundef nonnull %104, float* noundef nonnull %105, float noundef %147, float noundef %150, float noundef %121, float noundef %123, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #14
  %151 = load float, float* %15, align 4, !tbaa !56
  %152 = zext i32 %133 to i64
  %153 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %152
  %154 = load float, float* %153, align 4, !tbaa !56
  %155 = fadd float %151, %154
  store float %155, float* %153, align 4, !tbaa !56
  %156 = load float, float* %16, align 4, !tbaa !56
  %157 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %152
  %158 = load float, float* %157, align 4, !tbaa !56
  %159 = fadd float %156, %158
  store float %159, float* %157, align 4, !tbaa !56
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %107) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %106) #13
  br label %160

160:                                              ; preds = %136, %132
  %161 = add nuw nsw i32 %133, 1
  %162 = icmp eq i32 %161, 4
  br i1 %162, label %129, label %132, !llvm.loop !67

163:                                              ; preds = %207
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %102) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %101) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %100) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %99) #13
  br label %210

164:                                              ; preds = %207, %108
  %165 = phi i32 [ 0, %108 ], [ %208, %207 ]
  %166 = add nuw nsw i32 %165, %77
  %167 = icmp ult i32 %166, 6144
  br i1 %167, label %168, label %207

168:                                              ; preds = %164
  %169 = zext i32 %165 to i64
  %170 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %169
  %171 = load float, float* %170, align 4, !tbaa !56
  %172 = call fast float @air.simd_sum.f32(float %171) #15
  br i1 %109, label %173, label %188

173:                                              ; preds = %168
  %174 = fptrunc float %172 to bfloat
  %175 = bitcast float %172 to i32
  %176 = and i32 %175, 2139095040
  %177 = icmp eq i32 %176, 2139095040
  br i1 %177, label %183, label %178

178:                                              ; preds = %173
  %179 = fpext bfloat %174 to float
  %180 = bitcast float %179 to i32
  %181 = and i32 %180, 2139095040
  %182 = icmp eq i32 %181, 2139095040
  br i1 %182, label %183, label %185

183:                                              ; preds = %178, %173
  %184 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %110, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %185

185:                                              ; preds = %183, %178
  %186 = add nuw nsw i64 %113, %169
  %187 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %186
  store bfloat %174, bfloat addrspace(1)* %187, align 2, !tbaa !54
  br label %188

188:                                              ; preds = %185, %168
  %189 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %169
  %190 = load float, float* %189, align 4, !tbaa !56
  %191 = call fast float @air.simd_sum.f32(float %190) #15
  br i1 %109, label %192, label %207

192:                                              ; preds = %188
  %193 = fptrunc float %191 to bfloat
  %194 = bitcast float %191 to i32
  %195 = and i32 %194, 2139095040
  %196 = icmp eq i32 %195, 2139095040
  br i1 %196, label %202, label %197

197:                                              ; preds = %192
  %198 = fpext bfloat %193 to float
  %199 = bitcast float %198 to i32
  %200 = and i32 %199, 2139095040
  %201 = icmp eq i32 %200, 2139095040
  br i1 %201, label %202, label %204

202:                                              ; preds = %197, %192
  %203 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %110, i32 4, i32 0, i32 2, i32 0, i1 false) #11
  br label %204

204:                                              ; preds = %202, %197
  %205 = add nuw nsw i64 %115, %169
  %206 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %205
  store bfloat %193, bfloat addrspace(1)* %206, align 2, !tbaa !54
  br label %207

207:                                              ; preds = %204, %188, %164
  %208 = add nuw nsw i32 %165, 1
  %209 = icmp eq i32 %208, 4
  br i1 %209, label %163, label %164, !llvm.loop !68

210:                                              ; preds = %163, %89, %73, %70, %68
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i1 @air.any.v3i1(<3 x i1>) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture, i32, i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN18r5_raw_odd_literal12project_mathILt4ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #5 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #13
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !36
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %23, label %19

19:                                               ; preds = %11
  %20 = shl i32 %10, 4
  %21 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %32

23:                                               ; preds = %72, %11
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !42
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
  %42 = load bfloat, bfloat addrspace(1)* %41, align 2, !tbaa !54
  %43 = fpext bfloat %42 to float
  %44 = or i32 %38, 1
  %45 = zext i32 %44 to i64
  %46 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %45
  %47 = load bfloat, bfloat addrspace(1)* %46, align 2, !tbaa !54
  %48 = fpext bfloat %47 to float
  %49 = fadd float %43, %48
  %50 = or i32 %38, 2
  %51 = zext i32 %50 to i64
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %51
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !54
  %54 = fpext bfloat %53 to float
  %55 = fadd float %49, %54
  %56 = or i32 %38, 3
  %57 = zext i32 %56 to i64
  %58 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %57
  %59 = load bfloat, bfloat addrspace(1)* %58, align 2, !tbaa !54
  %60 = fpext bfloat %59 to float
  %61 = fadd float %55, %60
  %62 = fadd float %39, %61
  %63 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %40
  store float %43, float* %63, align 4, !tbaa !56
  %64 = fmul float %48, 6.250000e-02
  %65 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %45
  store float %64, float* %65, align 4, !tbaa !56
  %66 = fmul float %54, 3.906250e-03
  %67 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %51
  store float %66, float* %67, align 4, !tbaa !56
  %68 = fmul float %60, 0x3F30000000000000
  %69 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %57
  store float %68, float* %69, align 4, !tbaa !56
  %70 = add nuw nsw i32 %38, 4
  %71 = icmp ult i32 %38, 12
  br i1 %71, label %37, label %72, !llvm.loop !69

72:                                               ; preds = %37
  call void @_ZN18r5_raw_odd_literal16accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %34, i32 noundef %8, float* noundef nonnull %21, float noundef %62, i32 noundef 16, i1 noundef zeroext false, float* noundef nonnull %22) #14
  %73 = add i32 %33, 512
  %74 = icmp ult i32 %73, %17
  br i1 %74, label %32, label %23, !llvm.loop !70

75:                                               ; preds = %100
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #13
  ret void

76:                                               ; preds = %100, %23
  %77 = phi i32 [ 0, %23 ], [ %101, %100 ]
  %78 = add i32 %77, %7
  %79 = icmp ult i32 %78, %25
  br i1 %79, label %80, label %100

80:                                               ; preds = %76
  %81 = zext i32 %77 to i64
  %82 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %81
  %83 = load float, float* %82, align 4, !tbaa !56
  %84 = call fast float @air.simd_sum.f32(float %83) #15
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
  store bfloat %86, bfloat addrspace(1)* %99, align 2, !tbaa !54
  br label %100

100:                                              ; preds = %97, %80, %76
  %101 = add nuw nsw i32 %77, 1
  %102 = icmp eq i32 %101, 4
  br i1 %102, label %75, label %76, !llvm.loop !71
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1 immarg) #6

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #7 {
  br label %4

3:                                                ; preds = %4
  ret float %29

4:                                                ; preds = %4, %2
  %5 = phi i32 [ 0, %2 ], [ %37, %4 ]
  %6 = phi float [ 0.000000e+00, %2 ], [ %29, %4 ]
  %7 = zext i32 %5 to i64
  %8 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %7
  %9 = load bfloat, bfloat addrspace(1)* %8, align 2, !tbaa !54
  %10 = fpext bfloat %9 to float
  %11 = or i32 %5, 1
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %12
  %14 = load bfloat, bfloat addrspace(1)* %13, align 2, !tbaa !54
  %15 = fpext bfloat %14 to float
  %16 = fadd float %10, %15
  %17 = or i32 %5, 2
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %18
  %20 = load bfloat, bfloat addrspace(1)* %19, align 2, !tbaa !54
  %21 = fpext bfloat %20 to float
  %22 = fadd float %16, %21
  %23 = or i32 %5, 3
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %24
  %26 = load bfloat, bfloat addrspace(1)* %25, align 2, !tbaa !54
  %27 = fpext bfloat %26 to float
  %28 = fadd float %22, %27
  %29 = fadd float %6, %28
  %30 = getelementptr inbounds float, float* %1, i64 %7
  store float %10, float* %30, align 4, !tbaa !56
  %31 = fmul float %15, 6.250000e-02
  %32 = getelementptr inbounds float, float* %1, i64 %12
  store float %31, float* %32, align 4, !tbaa !56
  %33 = fmul float %21, 3.906250e-03
  %34 = getelementptr inbounds float, float* %1, i64 %18
  store float %33, float* %34, align 4, !tbaa !56
  %35 = fmul float %27, 0x3F30000000000000
  %36 = getelementptr inbounds float, float* %1, i64 %24
  store float %35, float* %36, align 4, !tbaa !56
  %37 = add nuw nsw i32 %5, 4
  %38 = icmp ult i32 %5, 12
  br i1 %38, label %4, label %3, !llvm.loop !69
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #7 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %59, float %11)
  store float %12, float* %7, align 4, !tbaa !56
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %72, float %13)
  store float %14, float* %8, align 4, !tbaa !56
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
  %36 = load float, float* %35, align 4, !tbaa !56
  %37 = zext i8 %29 to i32
  %38 = tail call float @air.convert.f.f32.s.i32(i32 %37) #10
  %39 = or i32 %33, 1
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds float, float* %1, i64 %40
  %42 = load float, float* %41, align 4, !tbaa !56
  %43 = zext i8 %30 to i32
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %43) #10
  %45 = fmul float %42, %44
  %46 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %45)
  %47 = or i32 %33, 2
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds float, float* %1, i64 %48
  %50 = load float, float* %49, align 4, !tbaa !56
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %31) #10
  %52 = tail call float @llvm.fmuladd.f32(float %50, float %51, float %46)
  %53 = or i32 %33, 3
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds float, float* %1, i64 %54
  %56 = load float, float* %55, align 4, !tbaa !56
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %32) #10
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %52)
  %59 = fadd float %16, %58
  %60 = getelementptr inbounds float, float* %2, i64 %34
  %61 = load float, float* %60, align 4, !tbaa !56
  %62 = getelementptr inbounds float, float* %2, i64 %40
  %63 = load float, float* %62, align 4, !tbaa !56
  %64 = fmul float %44, %63
  %65 = tail call float @llvm.fmuladd.f32(float %61, float %38, float %64)
  %66 = getelementptr inbounds float, float* %2, i64 %48
  %67 = load float, float* %66, align 4, !tbaa !56
  %68 = tail call float @llvm.fmuladd.f32(float %67, float %51, float %65)
  %69 = getelementptr inbounds float, float* %2, i64 %54
  %70 = load float, float* %69, align 4, !tbaa !56
  %71 = tail call float @llvm.fmuladd.f32(float %70, float %57, float %68)
  %72 = fadd float %17, %71
  %73 = add nuw nsw i32 %18, 1
  %74 = icmp eq i32 %73, 4
  br i1 %74, label %10, label %15, !llvm.loop !73
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN18r5_raw_odd_literal16accumulate_chunkILt4ELt64ELt16EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #7 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !51
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !74
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !42
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
  %50 = load bfloat, bfloat addrspace(1)* %49, align 2, !tbaa !54
  %51 = fpext bfloat %50 to float
  %52 = getelementptr inbounds bfloat, bfloat addrspace(1)* %48, i64 %30
  %53 = load bfloat, bfloat addrspace(1)* %52, align 2, !tbaa !54
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
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !72
  %63 = zext i8 %62 to i32
  %64 = or i32 %59, 1
  %65 = zext i32 %64 to i64
  %66 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %65
  %67 = load i8, i8 addrspace(1)* %66, align 1, !tbaa !72
  %68 = zext i8 %67 to i32
  %69 = shl nuw nsw i32 %68, 8
  %70 = shl nsw i32 %58, 2
  %71 = zext i32 %70 to i64
  %72 = getelementptr inbounds float, float* %7, i64 %71
  %73 = load float, float* %72, align 4, !tbaa !56
  %74 = and i32 %63, 15
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = or i32 %70, 1
  %77 = zext i32 %76 to i64
  %78 = getelementptr inbounds float, float* %7, i64 %77
  %79 = load float, float* %78, align 4, !tbaa !56
  %80 = and i32 %63, 240
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %80) #10
  %82 = fmul float %79, %81
  %83 = tail call float @llvm.fmuladd.f32(float %73, float %75, float %82) #13
  %84 = or i32 %70, 2
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds float, float* %7, i64 %85
  %87 = load float, float* %86, align 4, !tbaa !56
  %88 = and i32 %69, 3840
  %89 = tail call float @air.convert.f.f32.s.i32(i32 %88) #10
  %90 = tail call float @llvm.fmuladd.f32(float %87, float %89, float %83) #13
  %91 = or i32 %70, 3
  %92 = zext i32 %91 to i64
  %93 = getelementptr inbounds float, float* %7, i64 %92
  %94 = load float, float* %93, align 4, !tbaa !56
  %95 = and i32 %69, 61440
  %96 = tail call float @air.convert.f.f32.s.i32(i32 %95) #10
  %97 = tail call float @llvm.fmuladd.f32(float %94, float %96, float %90) #13
  %98 = fadd float %57, %97
  %99 = add nuw nsw i32 %58, 1
  %100 = icmp eq i32 %99, %31
  br i1 %100, label %101, label %56, !llvm.loop !75

101:                                              ; preds = %56, %55
  %102 = phi float [ 0.000000e+00, %55 ], [ %98, %56 ]
  %103 = fmul float %54, %8
  %104 = tail call float @llvm.fmuladd.f32(float %51, float %102, float %103) #13
  %105 = zext i32 %36 to i64
  %106 = getelementptr inbounds float, float* %11, i64 %105
  %107 = load float, float* %106, align 4, !tbaa !56
  %108 = fadd float %107, %104
  store float %108, float* %106, align 4, !tbaa !56
  br label %161

109:                                              ; preds = %109, %39
  %110 = phi float [ %151, %109 ], [ 0.000000e+00, %39 ]
  %111 = phi i32 [ %152, %109 ], [ 0, %39 ]
  %112 = shl nuw nsw i32 %111, 1
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %113
  %115 = load i8, i8 addrspace(1)* %114, align 1, !tbaa !72
  %116 = zext i8 %115 to i32
  %117 = or i32 %112, 1
  %118 = zext i32 %117 to i64
  %119 = getelementptr inbounds i8, i8 addrspace(1)* %42, i64 %118
  %120 = load i8, i8 addrspace(1)* %119, align 1, !tbaa !72
  %121 = zext i8 %120 to i32
  %122 = shl nuw nsw i32 %121, 8
  %123 = shl nuw nsw i32 %111, 2
  %124 = zext i32 %123 to i64
  %125 = getelementptr inbounds float, float* %7, i64 %124
  %126 = load float, float* %125, align 4, !tbaa !56
  %127 = and i32 %116, 15
  %128 = tail call float @air.convert.f.f32.s.i32(i32 %127) #10
  %129 = or i32 %123, 1
  %130 = zext i32 %129 to i64
  %131 = getelementptr inbounds float, float* %7, i64 %130
  %132 = load float, float* %131, align 4, !tbaa !56
  %133 = and i32 %116, 240
  %134 = tail call float @air.convert.f.f32.s.i32(i32 %133) #10
  %135 = fmul float %132, %134
  %136 = tail call float @llvm.fmuladd.f32(float %126, float %128, float %135) #13
  %137 = or i32 %123, 2
  %138 = zext i32 %137 to i64
  %139 = getelementptr inbounds float, float* %7, i64 %138
  %140 = load float, float* %139, align 4, !tbaa !56
  %141 = and i32 %122, 3840
  %142 = tail call float @air.convert.f.f32.s.i32(i32 %141) #10
  %143 = tail call float @llvm.fmuladd.f32(float %140, float %142, float %136) #13
  %144 = or i32 %123, 3
  %145 = zext i32 %144 to i64
  %146 = getelementptr inbounds float, float* %7, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !56
  %148 = and i32 %122, 61440
  %149 = tail call float @air.convert.f.f32.s.i32(i32 %148) #10
  %150 = tail call float @llvm.fmuladd.f32(float %147, float %149, float %143) #13
  %151 = fadd float %110, %150
  %152 = add nuw nsw i32 %111, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %154, label %109, !llvm.loop !76

154:                                              ; preds = %109
  %155 = fmul float %54, %8
  %156 = tail call float @llvm.fmuladd.f32(float %51, float %151, float %155) #13
  %157 = zext i32 %36 to i64
  %158 = getelementptr inbounds float, float* %11, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !56
  %160 = fadd float %159, %156
  store float %160, float* %158, align 4, !tbaa !56
  br label %161

161:                                              ; preds = %154, %101, %35
  %162 = add nuw nsw i32 %36, 1
  %163 = icmp eq i32 %162, 4
  br i1 %163, label %34, label %35, !llvm.loop !77
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare float @air.convert.f.f32.s.i32(i32) local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #8

; Function Attrs: convergent mustprogress nounwind willreturn
declare float @air.simd_sum.f32(float) local_unnamed_addr #9

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN18r5_raw_odd_literal12project_mathILt5ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #5 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #13
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !36
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %11
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !42
  br label %39

22:                                               ; preds = %11
  %23 = shl i32 %10, 4
  %24 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %27 = load i64, i64 addrspace(2)* %26, align 8, !tbaa !51
  %28 = mul i64 %27, %25
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %30 = load i64, i64 addrspace(2)* %29, align 8, !tbaa !74
  %31 = mul i64 %30, %25
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !42
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
  %58 = load bfloat, bfloat addrspace(1)* %57, align 2, !tbaa !54
  %59 = fpext bfloat %58 to float
  %60 = or i32 %54, 1
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %61
  %63 = load bfloat, bfloat addrspace(1)* %62, align 2, !tbaa !54
  %64 = fpext bfloat %63 to float
  %65 = fadd float %59, %64
  %66 = or i32 %54, 2
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %67
  %69 = load bfloat, bfloat addrspace(1)* %68, align 2, !tbaa !54
  %70 = fpext bfloat %69 to float
  %71 = fadd float %65, %70
  %72 = or i32 %54, 3
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %73
  %75 = load bfloat, bfloat addrspace(1)* %74, align 2, !tbaa !54
  %76 = fpext bfloat %75 to float
  %77 = fadd float %71, %76
  %78 = or i32 %54, 4
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %79
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !54
  %82 = fpext bfloat %81 to float
  %83 = fadd float %77, %82
  %84 = or i32 %54, 5
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %85
  %87 = load bfloat, bfloat addrspace(1)* %86, align 2, !tbaa !54
  %88 = fpext bfloat %87 to float
  %89 = fadd float %83, %88
  %90 = or i32 %54, 6
  %91 = zext i32 %90 to i64
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %91
  %93 = load bfloat, bfloat addrspace(1)* %92, align 2, !tbaa !54
  %94 = fpext bfloat %93 to float
  %95 = fadd float %89, %94
  %96 = or i32 %54, 7
  %97 = zext i32 %96 to i64
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %97
  %99 = load bfloat, bfloat addrspace(1)* %98, align 2, !tbaa !54
  %100 = fpext bfloat %99 to float
  %101 = fadd float %95, %100
  %102 = fadd float %55, %101
  %103 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %56
  store float %59, float* %103, align 4, !tbaa !56
  %104 = fmul float %64, 3.125000e-02
  %105 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %61
  store float %104, float* %105, align 4, !tbaa !56
  %106 = fmul float %70, 2.500000e-01
  %107 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %67
  store float %106, float* %107, align 4, !tbaa !56
  %108 = fmul float %76, 7.812500e-03
  %109 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %73
  store float %108, float* %109, align 4, !tbaa !56
  %110 = fmul float %82, 6.250000e-02
  %111 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %79
  store float %110, float* %111, align 4, !tbaa !56
  %112 = fmul float %88, 5.000000e-01
  %113 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %85
  store float %112, float* %113, align 4, !tbaa !56
  %114 = fmul float %94, 1.562500e-02
  %115 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %91
  store float %114, float* %115, align 4, !tbaa !56
  %116 = fmul float %100, 1.250000e-01
  %117 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %97
  store float %116, float* %117, align 4, !tbaa !56
  br i1 %53, label %52, label %118, !llvm.loop !78

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
  %139 = load bfloat, bfloat addrspace(1)* %138, align 2, !tbaa !54
  %140 = fpext bfloat %139 to float
  %141 = getelementptr inbounds bfloat, bfloat addrspace(1)* %137, i64 %122
  %142 = load bfloat, bfloat addrspace(1)* %141, align 2, !tbaa !54
  %143 = fpext bfloat %142 to float
  %144 = call fast float @_ZN18r5_raw_odd_literal23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %131, float* noundef nonnull %24, float noundef %140, float noundef %143, float noundef %102) #16
  %145 = zext i32 %125 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !56
  %148 = fadd float %144, %147
  store float %148, float* %146, align 4, !tbaa !56
  br label %149

149:                                              ; preds = %128, %124
  %150 = add nuw nsw i32 %125, 1
  %151 = icmp eq i32 %150, 4
  br i1 %151, label %152, label %124, !llvm.loop !79

152:                                              ; preds = %149
  %153 = add i32 %48, 512
  %154 = icmp ult i32 %153, %17
  br i1 %154, label %47, label %39, !llvm.loop !80

155:                                              ; preds = %180
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #13
  ret void

156:                                              ; preds = %180, %39
  %157 = phi i32 [ 0, %39 ], [ %181, %180 ]
  %158 = add i32 %157, %7
  %159 = icmp ult i32 %158, %40
  br i1 %159, label %160, label %180

160:                                              ; preds = %156
  %161 = zext i32 %157 to i64
  %162 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %161
  %163 = load float, float* %162, align 4, !tbaa !56
  %164 = call fast float @air.simd_sum.f32(float %163) #15
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
  store bfloat %166, bfloat addrspace(1)* %179, align 2, !tbaa !54
  br label %180

180:                                              ; preds = %177, %160, %156
  %181 = add nuw nsw i32 %157, 1
  %182 = icmp eq i32 %181, 4
  br i1 %182, label %155, label %156, !llvm.loop !81
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #7 {
  br label %4

3:                                                ; preds = %4
  ret float %54

4:                                                ; preds = %4, %2
  %5 = phi i1 [ true, %2 ], [ false, %4 ]
  %6 = phi i32 [ 0, %2 ], [ 8, %4 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %54, %4 ]
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %8
  %10 = load bfloat, bfloat addrspace(1)* %9, align 2, !tbaa !54
  %11 = fpext bfloat %10 to float
  %12 = or i32 %6, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %13
  %15 = load bfloat, bfloat addrspace(1)* %14, align 2, !tbaa !54
  %16 = fpext bfloat %15 to float
  %17 = fadd float %11, %16
  %18 = or i32 %6, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %19
  %21 = load bfloat, bfloat addrspace(1)* %20, align 2, !tbaa !54
  %22 = fpext bfloat %21 to float
  %23 = fadd float %17, %22
  %24 = or i32 %6, 3
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %25
  %27 = load bfloat, bfloat addrspace(1)* %26, align 2, !tbaa !54
  %28 = fpext bfloat %27 to float
  %29 = fadd float %23, %28
  %30 = or i32 %6, 4
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %31
  %33 = load bfloat, bfloat addrspace(1)* %32, align 2, !tbaa !54
  %34 = fpext bfloat %33 to float
  %35 = fadd float %29, %34
  %36 = or i32 %6, 5
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %37
  %39 = load bfloat, bfloat addrspace(1)* %38, align 2, !tbaa !54
  %40 = fpext bfloat %39 to float
  %41 = fadd float %35, %40
  %42 = or i32 %6, 6
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !54
  %46 = fpext bfloat %45 to float
  %47 = fadd float %41, %46
  %48 = or i32 %6, 7
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %49
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !54
  %52 = fpext bfloat %51 to float
  %53 = fadd float %47, %52
  %54 = fadd float %7, %53
  %55 = getelementptr inbounds float, float* %1, i64 %8
  store float %11, float* %55, align 4, !tbaa !56
  %56 = fmul float %16, 3.125000e-02
  %57 = getelementptr inbounds float, float* %1, i64 %13
  store float %56, float* %57, align 4, !tbaa !56
  %58 = fmul float %22, 2.500000e-01
  %59 = getelementptr inbounds float, float* %1, i64 %19
  store float %58, float* %59, align 4, !tbaa !56
  %60 = fmul float %28, 7.812500e-03
  %61 = getelementptr inbounds float, float* %1, i64 %25
  store float %60, float* %61, align 4, !tbaa !56
  %62 = fmul float %34, 6.250000e-02
  %63 = getelementptr inbounds float, float* %1, i64 %31
  store float %62, float* %63, align 4, !tbaa !56
  %64 = fmul float %40, 5.000000e-01
  %65 = getelementptr inbounds float, float* %1, i64 %37
  store float %64, float* %65, align 4, !tbaa !56
  %66 = fmul float %46, 1.562500e-02
  %67 = getelementptr inbounds float, float* %1, i64 %43
  store float %66, float* %67, align 4, !tbaa !56
  %68 = fmul float %52, 1.250000e-01
  %69 = getelementptr inbounds float, float* %1, i64 %49
  store float %68, float* %69, align 4, !tbaa !56
  br i1 %5, label %4, label %3, !llvm.loop !78
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #7 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %98, float %11)
  store float %12, float* %7, align 4, !tbaa !56
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %129, float %13)
  store float %14, float* %8, align 4, !tbaa !56
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
  %56 = tail call float @air.convert.f.f32.s.i32(i32 %32) #10
  %57 = load float, float* %25, align 4, !tbaa !56
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %19)
  %59 = tail call float @air.convert.f.f32.s.i32(i32 %33) #10
  %60 = getelementptr inbounds float, float* %25, i64 1
  %61 = load float, float* %60, align 4, !tbaa !56
  %62 = tail call float @llvm.fmuladd.f32(float %59, float %61, float %58)
  %63 = tail call float @air.convert.f.f32.s.i32(i32 %37) #10
  %64 = fmul float %61, 2.560000e+02
  %65 = tail call float @llvm.fmuladd.f32(float %63, float %64, float %62)
  %66 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %67 = getelementptr inbounds float, float* %25, i64 2
  %68 = load float, float* %67, align 4, !tbaa !56
  %69 = tail call float @llvm.fmuladd.f32(float %66, float %68, float %65)
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %39) #10
  %71 = getelementptr inbounds float, float* %25, i64 3
  %72 = load float, float* %71, align 4, !tbaa !56
  %73 = tail call float @llvm.fmuladd.f32(float %70, float %72, float %69)
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %43) #10
  %75 = fmul float %72, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %74, float %75, float %73)
  %77 = tail call float @air.convert.f.f32.s.i32(i32 %44) #10
  %78 = getelementptr inbounds float, float* %25, i64 4
  %79 = load float, float* %78, align 4, !tbaa !56
  %80 = tail call float @llvm.fmuladd.f32(float %77, float %79, float %76)
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %48) #10
  %82 = fmul float %79, 2.560000e+02
  %83 = tail call float @llvm.fmuladd.f32(float %81, float %82, float %80)
  %84 = tail call float @air.convert.f.f32.s.i32(i32 %49) #10
  %85 = getelementptr inbounds float, float* %25, i64 5
  %86 = load float, float* %85, align 4, !tbaa !56
  %87 = tail call float @llvm.fmuladd.f32(float %84, float %86, float %83)
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %50) #10
  %89 = getelementptr inbounds float, float* %25, i64 6
  %90 = load float, float* %89, align 4, !tbaa !56
  %91 = tail call float @llvm.fmuladd.f32(float %88, float %90, float %87)
  %92 = tail call float @air.convert.f.f32.s.i32(i32 %54) #10
  %93 = fmul float %90, 2.560000e+02
  %94 = tail call float @llvm.fmuladd.f32(float %92, float %93, float %91)
  %95 = tail call float @air.convert.f.f32.s.i32(i32 %55) #10
  %96 = getelementptr inbounds float, float* %25, i64 7
  %97 = load float, float* %96, align 4, !tbaa !56
  %98 = tail call float @llvm.fmuladd.f32(float %95, float %97, float %94)
  %99 = load float, float* %26, align 4, !tbaa !56
  %100 = tail call float @llvm.fmuladd.f32(float %56, float %99, float %20)
  %101 = getelementptr inbounds float, float* %26, i64 1
  %102 = load float, float* %101, align 4, !tbaa !56
  %103 = tail call float @llvm.fmuladd.f32(float %59, float %102, float %100)
  %104 = fmul float %102, 2.560000e+02
  %105 = tail call float @llvm.fmuladd.f32(float %63, float %104, float %103)
  %106 = getelementptr inbounds float, float* %26, i64 2
  %107 = load float, float* %106, align 4, !tbaa !56
  %108 = tail call float @llvm.fmuladd.f32(float %66, float %107, float %105)
  %109 = getelementptr inbounds float, float* %26, i64 3
  %110 = load float, float* %109, align 4, !tbaa !56
  %111 = tail call float @llvm.fmuladd.f32(float %70, float %110, float %108)
  %112 = fmul float %110, 2.560000e+02
  %113 = tail call float @llvm.fmuladd.f32(float %74, float %112, float %111)
  %114 = getelementptr inbounds float, float* %26, i64 4
  %115 = load float, float* %114, align 4, !tbaa !56
  %116 = tail call float @llvm.fmuladd.f32(float %77, float %115, float %113)
  %117 = fmul float %115, 2.560000e+02
  %118 = tail call float @llvm.fmuladd.f32(float %81, float %117, float %116)
  %119 = getelementptr inbounds float, float* %26, i64 5
  %120 = load float, float* %119, align 4, !tbaa !56
  %121 = tail call float @llvm.fmuladd.f32(float %84, float %120, float %118)
  %122 = getelementptr inbounds float, float* %26, i64 6
  %123 = load float, float* %122, align 4, !tbaa !56
  %124 = tail call float @llvm.fmuladd.f32(float %88, float %123, float %121)
  %125 = fmul float %123, 2.560000e+02
  %126 = tail call float @llvm.fmuladd.f32(float %92, float %125, float %124)
  %127 = getelementptr inbounds float, float* %26, i64 7
  %128 = load float, float* %127, align 4, !tbaa !56
  %129 = tail call float @llvm.fmuladd.f32(float %95, float %128, float %126)
  br i1 %21, label %15, label %10, !llvm.loop !82
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4) local_unnamed_addr #7 {
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
  %21 = load i8, i8 addrspace(1)* %20, align 1, !tbaa !72
  %22 = zext i8 %21 to i32
  %23 = and i32 %22, 31
  %24 = tail call float @air.convert.f.f32.s.i32(i32 %23) #10
  %25 = load float, float* %17, align 4, !tbaa !56
  %26 = tail call float @llvm.fmuladd.f32(float %24, float %25, float %12)
  %27 = and i32 %22, 224
  %28 = tail call float @air.convert.f.f32.s.i32(i32 %27) #10
  %29 = getelementptr inbounds float, float* %17, i64 1
  %30 = load float, float* %29, align 4, !tbaa !56
  %31 = tail call float @llvm.fmuladd.f32(float %28, float %30, float %26)
  %32 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 1
  %33 = load i8, i8 addrspace(1)* %32, align 1, !tbaa !72
  %34 = zext i8 %33 to i32
  %35 = and i32 %34, 3
  %36 = tail call float @air.convert.f.f32.s.i32(i32 %35) #10
  %37 = fmul float %30, 2.560000e+02
  %38 = tail call float @llvm.fmuladd.f32(float %36, float %37, float %31)
  %39 = and i32 %34, 124
  %40 = tail call float @air.convert.f.f32.s.i32(i32 %39) #10
  %41 = getelementptr inbounds float, float* %17, i64 2
  %42 = load float, float* %41, align 4, !tbaa !56
  %43 = tail call float @llvm.fmuladd.f32(float %40, float %42, float %38)
  %44 = and i32 %34, 128
  %45 = tail call float @air.convert.f.f32.s.i32(i32 %44) #10
  %46 = getelementptr inbounds float, float* %17, i64 3
  %47 = load float, float* %46, align 4, !tbaa !56
  %48 = tail call float @llvm.fmuladd.f32(float %45, float %47, float %43)
  %49 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 2
  %50 = load i8, i8 addrspace(1)* %49, align 1, !tbaa !72
  %51 = zext i8 %50 to i32
  %52 = and i32 %51, 15
  %53 = tail call float @air.convert.f.f32.s.i32(i32 %52) #10
  %54 = fmul float %47, 2.560000e+02
  %55 = tail call float @llvm.fmuladd.f32(float %53, float %54, float %48)
  %56 = and i32 %51, 240
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %56) #10
  %58 = getelementptr inbounds float, float* %17, i64 4
  %59 = load float, float* %58, align 4, !tbaa !56
  %60 = tail call float @llvm.fmuladd.f32(float %57, float %59, float %55)
  %61 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 3
  %62 = load i8, i8 addrspace(1)* %61, align 1, !tbaa !72
  %63 = zext i8 %62 to i32
  %64 = and i32 %63, 1
  %65 = tail call float @air.convert.f.f32.s.i32(i32 %64) #10
  %66 = fmul float %59, 2.560000e+02
  %67 = tail call float @llvm.fmuladd.f32(float %65, float %66, float %60)
  %68 = and i32 %63, 62
  %69 = tail call float @air.convert.f.f32.s.i32(i32 %68) #10
  %70 = getelementptr inbounds float, float* %17, i64 5
  %71 = load float, float* %70, align 4, !tbaa !56
  %72 = tail call float @llvm.fmuladd.f32(float %69, float %71, float %67)
  %73 = and i32 %63, 192
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %73) #10
  %75 = getelementptr inbounds float, float* %17, i64 6
  %76 = load float, float* %75, align 4, !tbaa !56
  %77 = tail call float @llvm.fmuladd.f32(float %74, float %76, float %72)
  %78 = getelementptr inbounds i8, i8 addrspace(1)* %20, i64 4
  %79 = load i8, i8 addrspace(1)* %78, align 1, !tbaa !72
  %80 = zext i8 %79 to i32
  %81 = and i32 %80, 7
  %82 = tail call float @air.convert.f.f32.s.i32(i32 %81) #10
  %83 = fmul float %76, 2.560000e+02
  %84 = tail call float @llvm.fmuladd.f32(float %82, float %83, float %77)
  %85 = and i32 %80, 248
  %86 = tail call float @air.convert.f.f32.s.i32(i32 %85) #10
  %87 = getelementptr inbounds float, float* %17, i64 7
  %88 = load float, float* %87, align 4, !tbaa !56
  %89 = tail call float @llvm.fmuladd.f32(float %86, float %88, float %84)
  br i1 %10, label %9, label %6, !llvm.loop !83
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN18r5_raw_odd_literal12project_mathILt5ELt128ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #5 {
  %12 = alloca [16 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %14) #13
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !36
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %11
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !42
  br label %39

22:                                               ; preds = %11
  %23 = shl i32 %10, 4
  %24 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %25 = zext i32 %8 to i64
  %26 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %27 = load i64, i64 addrspace(2)* %26, align 8, !tbaa !51
  %28 = mul i64 %27, %25
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 9
  %30 = load i64, i64 addrspace(2)* %29, align 8, !tbaa !74
  %31 = mul i64 %30, %25
  %32 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %33 = load i32, i32 addrspace(2)* %32, align 4, !tbaa !42
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
  %58 = load bfloat, bfloat addrspace(1)* %57, align 2, !tbaa !54
  %59 = fpext bfloat %58 to float
  %60 = or i32 %54, 1
  %61 = zext i32 %60 to i64
  %62 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %61
  %63 = load bfloat, bfloat addrspace(1)* %62, align 2, !tbaa !54
  %64 = fpext bfloat %63 to float
  %65 = fadd float %59, %64
  %66 = or i32 %54, 2
  %67 = zext i32 %66 to i64
  %68 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %67
  %69 = load bfloat, bfloat addrspace(1)* %68, align 2, !tbaa !54
  %70 = fpext bfloat %69 to float
  %71 = fadd float %65, %70
  %72 = or i32 %54, 3
  %73 = zext i32 %72 to i64
  %74 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %73
  %75 = load bfloat, bfloat addrspace(1)* %74, align 2, !tbaa !54
  %76 = fpext bfloat %75 to float
  %77 = fadd float %71, %76
  %78 = or i32 %54, 4
  %79 = zext i32 %78 to i64
  %80 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %79
  %81 = load bfloat, bfloat addrspace(1)* %80, align 2, !tbaa !54
  %82 = fpext bfloat %81 to float
  %83 = fadd float %77, %82
  %84 = or i32 %54, 5
  %85 = zext i32 %84 to i64
  %86 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %85
  %87 = load bfloat, bfloat addrspace(1)* %86, align 2, !tbaa !54
  %88 = fpext bfloat %87 to float
  %89 = fadd float %83, %88
  %90 = or i32 %54, 6
  %91 = zext i32 %90 to i64
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %91
  %93 = load bfloat, bfloat addrspace(1)* %92, align 2, !tbaa !54
  %94 = fpext bfloat %93 to float
  %95 = fadd float %89, %94
  %96 = or i32 %54, 7
  %97 = zext i32 %96 to i64
  %98 = getelementptr inbounds bfloat, bfloat addrspace(1)* %51, i64 %97
  %99 = load bfloat, bfloat addrspace(1)* %98, align 2, !tbaa !54
  %100 = fpext bfloat %99 to float
  %101 = fadd float %95, %100
  %102 = fadd float %55, %101
  %103 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %56
  store float %59, float* %103, align 4, !tbaa !56
  %104 = fmul float %64, 3.125000e-02
  %105 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %61
  store float %104, float* %105, align 4, !tbaa !56
  %106 = fmul float %70, 2.500000e-01
  %107 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %67
  store float %106, float* %107, align 4, !tbaa !56
  %108 = fmul float %76, 7.812500e-03
  %109 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %73
  store float %108, float* %109, align 4, !tbaa !56
  %110 = fmul float %82, 6.250000e-02
  %111 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %79
  store float %110, float* %111, align 4, !tbaa !56
  %112 = fmul float %88, 5.000000e-01
  %113 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %85
  store float %112, float* %113, align 4, !tbaa !56
  %114 = fmul float %94, 1.562500e-02
  %115 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %91
  store float %114, float* %115, align 4, !tbaa !56
  %116 = fmul float %100, 1.250000e-01
  %117 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 %97
  store float %116, float* %117, align 4, !tbaa !56
  br i1 %53, label %52, label %118, !llvm.loop !78

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
  %139 = load bfloat, bfloat addrspace(1)* %138, align 2, !tbaa !54
  %140 = fpext bfloat %139 to float
  %141 = getelementptr inbounds bfloat, bfloat addrspace(1)* %137, i64 %122
  %142 = load bfloat, bfloat addrspace(1)* %141, align 2, !tbaa !54
  %143 = fpext bfloat %142 to float
  %144 = call fast float @_ZN18r5_raw_odd_literal23mlx_qmv_f32xsum_v1_qdotIfLi16ELi5EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_(i8 addrspace(1)* noundef %131, float* noundef nonnull %24, float noundef %140, float noundef %143, float noundef %102) #16
  %145 = zext i32 %125 to i64
  %146 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %145
  %147 = load float, float* %146, align 4, !tbaa !56
  %148 = fadd float %144, %147
  store float %148, float* %146, align 4, !tbaa !56
  br label %149

149:                                              ; preds = %128, %124
  %150 = add nuw nsw i32 %125, 1
  %151 = icmp eq i32 %150, 4
  br i1 %151, label %152, label %124, !llvm.loop !84

152:                                              ; preds = %149
  %153 = add i32 %48, 512
  %154 = icmp ult i32 %153, %17
  br i1 %154, label %47, label %39, !llvm.loop !85

155:                                              ; preds = %180
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %14) #13
  ret void

156:                                              ; preds = %180, %39
  %157 = phi i32 [ 0, %39 ], [ %181, %180 ]
  %158 = add i32 %157, %7
  %159 = icmp ult i32 %158, %40
  br i1 %159, label %160, label %180

160:                                              ; preds = %156
  %161 = zext i32 %157 to i64
  %162 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %161
  %163 = load float, float* %162, align 4, !tbaa !56
  %164 = call fast float @air.simd_sum.f32(float %163) #15
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
  store bfloat %166, bfloat addrspace(1)* %179, align 2, !tbaa !54
  br label %180

180:                                              ; preds = %177, %160, %156
  %181 = add nuw nsw i32 %157, 1
  %182 = icmp eq i32 %181, 4
  br i1 %182, label %155, label %156, !llvm.loop !86
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN18r5_raw_odd_literal12project_mathILt6ELt64ELb1EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsjjmj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %4, %"struct.metal::_atomic" addrspace(1)* noundef %5, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %8, i64 noundef %9, i32 noundef %10) local_unnamed_addr #5 {
  %12 = alloca [8 x float], align 4
  %13 = alloca [4 x float], align 4
  %14 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %14) #13
  %15 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %15, i8 0, i64 16, i1 false)
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 2
  %17 = load i32, i32 addrspace(2)* %16, align 8, !tbaa !36
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %23, label %19

19:                                               ; preds = %11
  %20 = shl i32 %10, 3
  %21 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %22 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 0
  br label %32

23:                                               ; preds = %71, %11
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %25 = load i32, i32 addrspace(2)* %24, align 4, !tbaa !42
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
  %43 = load bfloat, bfloat addrspace(1)* %42, align 2, !tbaa !54
  %44 = fpext bfloat %43 to float
  %45 = or i32 %39, 1
  %46 = zext i32 %45 to i64
  %47 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %46
  %48 = load bfloat, bfloat addrspace(1)* %47, align 2, !tbaa !54
  %49 = fpext bfloat %48 to float
  %50 = fadd float %44, %49
  %51 = or i32 %39, 2
  %52 = zext i32 %51 to i64
  %53 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %52
  %54 = load bfloat, bfloat addrspace(1)* %53, align 2, !tbaa !54
  %55 = fpext bfloat %54 to float
  %56 = fadd float %50, %55
  %57 = or i32 %39, 3
  %58 = zext i32 %57 to i64
  %59 = getelementptr inbounds bfloat, bfloat addrspace(1)* %36, i64 %58
  %60 = load bfloat, bfloat addrspace(1)* %59, align 2, !tbaa !54
  %61 = fpext bfloat %60 to float
  %62 = fadd float %56, %61
  %63 = fadd float %40, %62
  %64 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %41
  store float %44, float* %64, align 4, !tbaa !56
  %65 = fmul float %49, 1.562500e-02
  %66 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %46
  store float %65, float* %66, align 4, !tbaa !56
  %67 = fmul float %55, 6.250000e-02
  %68 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %52
  store float %67, float* %68, align 4, !tbaa !56
  %69 = fmul float %61, 2.500000e-01
  %70 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 %58
  store float %69, float* %70, align 4, !tbaa !56
  br i1 %38, label %37, label %71, !llvm.loop !87

71:                                               ; preds = %37
  call void @_ZN18r5_raw_odd_literal16accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6, i32 noundef %7, i32 noundef %34, i32 noundef %8, float* noundef nonnull %21, float noundef %63, i32 noundef 8, i1 noundef zeroext false, float* noundef nonnull %22) #12
  %72 = add i32 %33, 256
  %73 = icmp ult i32 %72, %17
  br i1 %73, label %32, label %23, !llvm.loop !88

74:                                               ; preds = %99
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %15) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %14) #13
  ret void

75:                                               ; preds = %99, %23
  %76 = phi i32 [ 0, %23 ], [ %100, %99 ]
  %77 = add i32 %76, %7
  %78 = icmp ult i32 %77, %25
  br i1 %78, label %79, label %99

79:                                               ; preds = %75
  %80 = zext i32 %76 to i64
  %81 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %80
  %82 = load float, float* %81, align 4, !tbaa !56
  %83 = call fast float @air.simd_sum.f32(float %82) #15
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
  store bfloat %85, bfloat addrspace(1)* %98, align 2, !tbaa !54
  br label %99

99:                                               ; preds = %96, %79, %75
  %100 = add nuw nsw i32 %76, 1
  %101 = icmp eq i32 %100, 4
  br i1 %101, label %74, label %75, !llvm.loop !89
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #7 {
  br label %4

3:                                                ; preds = %4
  ret float %30

4:                                                ; preds = %4, %2
  %5 = phi i1 [ true, %2 ], [ false, %4 ]
  %6 = phi i32 [ 0, %2 ], [ 4, %4 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %30, %4 ]
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %8
  %10 = load bfloat, bfloat addrspace(1)* %9, align 2, !tbaa !54
  %11 = fpext bfloat %10 to float
  %12 = or i32 %6, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %13
  %15 = load bfloat, bfloat addrspace(1)* %14, align 2, !tbaa !54
  %16 = fpext bfloat %15 to float
  %17 = fadd float %11, %16
  %18 = or i32 %6, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %19
  %21 = load bfloat, bfloat addrspace(1)* %20, align 2, !tbaa !54
  %22 = fpext bfloat %21 to float
  %23 = fadd float %17, %22
  %24 = or i32 %6, 3
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %25
  %27 = load bfloat, bfloat addrspace(1)* %26, align 2, !tbaa !54
  %28 = fpext bfloat %27 to float
  %29 = fadd float %23, %28
  %30 = fadd float %7, %29
  %31 = getelementptr inbounds float, float* %1, i64 %8
  store float %11, float* %31, align 4, !tbaa !56
  %32 = fmul float %16, 1.562500e-02
  %33 = getelementptr inbounds float, float* %1, i64 %13
  store float %32, float* %33, align 4, !tbaa !56
  %34 = fmul float %22, 6.250000e-02
  %35 = getelementptr inbounds float, float* %1, i64 %19
  store float %34, float* %35, align 4, !tbaa !56
  %36 = fmul float %28, 2.500000e-01
  %37 = getelementptr inbounds float, float* %1, i64 %25
  store float %36, float* %37, align 4, !tbaa !56
  br i1 %5, label %4, label %3, !llvm.loop !87
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #7 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %64, float %11)
  store float %12, float* %7, align 4, !tbaa !56
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %79, float %13)
  store float %14, float* %8, align 4, !tbaa !56
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
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %32) #10
  %45 = load float, float* %25, align 4, !tbaa !56
  %46 = tail call float @llvm.fmuladd.f32(float %44, float %45, float %19)
  %47 = tail call float @air.convert.f.f32.s.i32(i32 %33) #10
  %48 = getelementptr inbounds float, float* %25, i64 1
  %49 = load float, float* %48, align 4, !tbaa !56
  %50 = tail call float @llvm.fmuladd.f32(float %47, float %49, float %46)
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %37) #10
  %52 = fmul float %49, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %50)
  %54 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %55 = getelementptr inbounds float, float* %25, i64 2
  %56 = load float, float* %55, align 4, !tbaa !56
  %57 = tail call float @llvm.fmuladd.f32(float %54, float %56, float %53)
  %58 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %59 = fmul float %56, 2.560000e+02
  %60 = tail call float @llvm.fmuladd.f32(float %58, float %59, float %57)
  %61 = tail call float @air.convert.f.f32.s.i32(i32 %43) #10
  %62 = getelementptr inbounds float, float* %25, i64 3
  %63 = load float, float* %62, align 4, !tbaa !56
  %64 = tail call float @llvm.fmuladd.f32(float %61, float %63, float %60)
  %65 = load float, float* %26, align 4, !tbaa !56
  %66 = tail call float @llvm.fmuladd.f32(float %44, float %65, float %20)
  %67 = getelementptr inbounds float, float* %26, i64 1
  %68 = load float, float* %67, align 4, !tbaa !56
  %69 = tail call float @llvm.fmuladd.f32(float %47, float %68, float %66)
  %70 = fmul float %68, 2.560000e+02
  %71 = tail call float @llvm.fmuladd.f32(float %51, float %70, float %69)
  %72 = getelementptr inbounds float, float* %26, i64 2
  %73 = load float, float* %72, align 4, !tbaa !56
  %74 = tail call float @llvm.fmuladd.f32(float %54, float %73, float %71)
  %75 = fmul float %73, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %58, float %75, float %74)
  %77 = getelementptr inbounds float, float* %26, i64 3
  %78 = load float, float* %77, align 4, !tbaa !56
  %79 = tail call float @llvm.fmuladd.f32(float %61, float %78, float %76)
  br i1 %21, label %15, label %10, !llvm.loop !90
}

; Function Attrs: convergent inlinehint mustprogress nounwind
define linkonce_odr void @_ZN18r5_raw_odd_literal16accumulate_chunkILt6ELt64ELt8EEEvPU9MTLdeviceKhS2_S2_RU11MTLconstantK17FlashAffineParamsjjjPU9MTLthreadKffjbPU9MTLthreadf(i8 addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, float* noundef %7, float noundef %8, i32 noundef %9, i1 noundef zeroext %10, float* noundef %11) local_unnamed_addr #5 {
  %13 = zext i32 %6 to i64
  %14 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 11
  %15 = load i64, i64 addrspace(2)* %14, align 8, !tbaa !51
  %16 = mul i64 %15, %13
  %17 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 9
  %18 = load i64, i64 addrspace(2)* %17, align 8, !tbaa !74
  %19 = mul i64 %18, %13
  %20 = lshr i32 %5, 6
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %3, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !42
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
  %49 = load bfloat, bfloat addrspace(1)* %48, align 2, !tbaa !54
  %50 = fpext bfloat %49 to float
  %51 = getelementptr inbounds bfloat, bfloat addrspace(1)* %47, i64 %31
  %52 = load bfloat, bfloat addrspace(1)* %51, align 2, !tbaa !54
  %53 = fpext bfloat %52 to float
  br i1 %10, label %54, label %60

54:                                               ; preds = %38
  %55 = tail call fast float @_ZN18r5_raw_odd_literal28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %41, float* noundef %7, float noundef %50, float noundef %53, float noundef %8, i32 noundef %9) #14
  %56 = zext i32 %35 to i64
  %57 = getelementptr inbounds float, float* %11, i64 %56
  %58 = load float, float* %57, align 4, !tbaa !56
  %59 = fadd float %55, %58
  store float %59, float* %57, align 4, !tbaa !56
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
  %72 = load i8, i8 addrspace(1)* %71, align 1, !tbaa !72
  %73 = zext i8 %72 to i32
  %74 = and i32 %73, 63
  %75 = tail call float @air.convert.f.f32.s.i32(i32 %74) #10
  %76 = load float, float* %68, align 4, !tbaa !56
  %77 = tail call float @llvm.fmuladd.f32(float %75, float %76, float %63) #13
  %78 = and i32 %73, 192
  %79 = tail call float @air.convert.f.f32.s.i32(i32 %78) #10
  %80 = getelementptr inbounds float, float* %68, i64 1
  %81 = load float, float* %80, align 4, !tbaa !56
  %82 = tail call float @llvm.fmuladd.f32(float %79, float %81, float %77) #13
  %83 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 1
  %84 = load i8, i8 addrspace(1)* %83, align 1, !tbaa !72
  %85 = zext i8 %84 to i32
  %86 = and i32 %85, 15
  %87 = tail call float @air.convert.f.f32.s.i32(i32 %86) #10
  %88 = fmul float %81, 2.560000e+02
  %89 = tail call float @llvm.fmuladd.f32(float %87, float %88, float %82) #13
  %90 = and i32 %85, 240
  %91 = tail call float @air.convert.f.f32.s.i32(i32 %90) #10
  %92 = getelementptr inbounds float, float* %68, i64 2
  %93 = load float, float* %92, align 4, !tbaa !56
  %94 = tail call float @llvm.fmuladd.f32(float %91, float %93, float %89) #13
  %95 = getelementptr inbounds i8, i8 addrspace(1)* %71, i64 2
  %96 = load i8, i8 addrspace(1)* %95, align 1, !tbaa !72
  %97 = zext i8 %96 to i32
  %98 = and i32 %97, 3
  %99 = tail call float @air.convert.f.f32.s.i32(i32 %98) #10
  %100 = fmul float %93, 2.560000e+02
  %101 = tail call float @llvm.fmuladd.f32(float %99, float %100, float %94) #13
  %102 = and i32 %97, 252
  %103 = tail call float @air.convert.f.f32.s.i32(i32 %102) #10
  %104 = getelementptr inbounds float, float* %68, i64 3
  %105 = load float, float* %104, align 4, !tbaa !56
  %106 = tail call float @llvm.fmuladd.f32(float %103, float %105, float %101) #13
  br i1 %61, label %60, label %107, !llvm.loop !91

107:                                              ; preds = %60
  %108 = fmul float %53, %8
  %109 = tail call float @llvm.fmuladd.f32(float %50, float %106, float %108) #13
  %110 = zext i32 %35 to i64
  %111 = getelementptr inbounds float, float* %11, i64 %110
  %112 = load float, float* %111, align 4, !tbaa !56
  %113 = fadd float %112, %109
  store float %113, float* %111, align 4, !tbaa !56
  br label %114

114:                                              ; preds = %107, %54, %34
  %115 = add nuw nsw i32 %35, 1
  %116 = icmp eq i32 %115, 4
  br i1 %116, label %33, label %34, !llvm.loop !92
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal28mlx_qmv_f32xsum_v1_qdot_safeIfLi8ELi6EEET_PU9MTLdeviceKhPU9MTLthreadKS1_S1_S1_S1_i(i8 addrspace(1)* noundef %0, float* noundef %1, float noundef %2, float noundef %3, float noundef %4, i32 noundef %5) local_unnamed_addr #7 {
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
  %24 = load i8, i8 addrspace(1)* %23, align 1, !tbaa !72
  %25 = zext i8 %24 to i32
  %26 = and i32 %25, 63
  %27 = tail call float @air.convert.f.f32.s.i32(i32 %26) #10
  %28 = load float, float* %20, align 4, !tbaa !56
  %29 = tail call float @llvm.fmuladd.f32(float %27, float %28, float %15)
  %30 = and i32 %25, 192
  %31 = tail call float @air.convert.f.f32.s.i32(i32 %30) #10
  %32 = getelementptr inbounds float, float* %20, i64 1
  %33 = load float, float* %32, align 4, !tbaa !56
  %34 = tail call float @llvm.fmuladd.f32(float %31, float %33, float %29)
  %35 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 1
  %36 = load i8, i8 addrspace(1)* %35, align 1, !tbaa !72
  %37 = zext i8 %36 to i32
  %38 = and i32 %37, 15
  %39 = tail call float @air.convert.f.f32.s.i32(i32 %38) #10
  %40 = fmul float %33, 2.560000e+02
  %41 = tail call float @llvm.fmuladd.f32(float %39, float %40, float %34)
  %42 = and i32 %37, 240
  %43 = tail call float @air.convert.f.f32.s.i32(i32 %42) #10
  %44 = getelementptr inbounds float, float* %20, i64 2
  %45 = load float, float* %44, align 4, !tbaa !56
  %46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %41)
  %47 = getelementptr inbounds i8, i8 addrspace(1)* %23, i64 2
  %48 = load i8, i8 addrspace(1)* %47, align 1, !tbaa !72
  %49 = zext i8 %48 to i32
  %50 = and i32 %49, 3
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %50) #10
  %52 = fmul float %45, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %46)
  %54 = and i32 %49, 252
  %55 = tail call float @air.convert.f.f32.s.i32(i32 %54) #10
  %56 = getelementptr inbounds float, float* %20, i64 3
  %57 = load float, float* %56, align 4, !tbaa !56
  %58 = tail call float @llvm.fmuladd.f32(float %55, float %57, float %53)
  %59 = add nuw nsw i32 %14, 1
  %60 = icmp eq i32 %59, %7
  br i1 %60, label %9, label %13, !llvm.loop !93
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
!air.kernel = !{!9, !26, !27, !28}
!air.compile_options = !{!29, !30, !31}
!llvm.ident = !{!32}
!air.version = !{!33}
!air.language_version = !{!34}
!air.source_file_name = !{!35}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32)* @r5_raw_odd_rowpair_sep22_timed_q4_g64, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !20, !22, !23, !24, !25}
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
!22 = !{i32 8, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"group"}
!23 = !{i32 9, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint3", !"air.arg_name", !"threads"}
!24 = !{i32 10, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
!25 = !{i32 11, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!26 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32)* @r5_raw_odd_rowpair_sep22_timed_q5_g64, !10, !11}
!27 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32)* @r5_raw_odd_rowpair_sep22_timed_q5_g128, !10, !11}
!28 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32)* @r5_raw_odd_rowpair_sep22_timed_q6_g64, !10, !11}
!29 = !{!"air.compile.denorms_disable"}
!30 = !{!"air.compile.fast_math_enable"}
!31 = !{!"air.compile.framebuffer_fetch_enable"}
!32 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!33 = !{i32 2, i32 9, i32 0}
!34 = !{!"Metal", i32 4, i32 1, i32 0}
!35 = !{!"/Users/mweinbach/Projects/splash/dev/benchmarks/R5_raw_odd_rowpair_sep22/kernel/candidate.metal"}
!36 = !{!37, !38, i64 8}
!37 = !{!"_ZTS17FlashAffineParams", !38, i64 0, !38, i64 4, !38, i64 8, !38, i64 12, !38, i64 16, !38, i64 20, !38, i64 24, !38, i64 28, !41, i64 32, !41, i64 40, !41, i64 48, !41, i64 56}
!38 = !{!"int", !39, i64 0}
!39 = !{!"omnipotent char", !40, i64 0}
!40 = !{!"Simple C++ TBAA"}
!41 = !{!"long", !39, i64 0}
!42 = !{!37, !38, i64 12}
!43 = !{!37, !38, i64 0}
!44 = !{!37, !38, i64 4}
!45 = !{!37, !38, i64 16}
!46 = !{!37, !38, i64 28}
!47 = !{!37, !38, i64 20}
!48 = !{!37, !38, i64 24}
!49 = !{!37, !41, i64 32}
!50 = !{!37, !41, i64 48}
!51 = !{!37, !41, i64 56}
!52 = distinct !{!52, !53}
!53 = !{!"llvm.loop.mustprogress"}
!54 = !{!55, !55, i64 0}
!55 = !{!"bfloat", !39, i64 0}
!56 = !{!57, !57, i64 0}
!57 = !{!"float", !39, i64 0}
!58 = distinct !{!58, !53}
!59 = distinct !{!59, !53}
!60 = distinct !{!60, !53}
!61 = distinct !{!61, !53}
!62 = distinct !{!62, !53}
!63 = distinct !{!63, !53}
!64 = distinct !{!64, !53}
!65 = distinct !{!65, !53}
!66 = distinct !{!66, !53}
!67 = distinct !{!67, !53}
!68 = distinct !{!68, !53}
!69 = distinct !{!69, !53}
!70 = distinct !{!70, !53}
!71 = distinct !{!71, !53}
!72 = !{!39, !39, i64 0}
!73 = distinct !{!73, !53}
!74 = !{!37, !41, i64 40}
!75 = distinct !{!75, !53}
!76 = distinct !{!76, !53}
!77 = distinct !{!77, !53}
!78 = distinct !{!78, !53}
!79 = distinct !{!79, !53}
!80 = distinct !{!80, !53}
!81 = distinct !{!81, !53}
!82 = distinct !{!82, !53}
!83 = distinct !{!83, !53}
!84 = distinct !{!84, !53}
!85 = distinct !{!85, !53}
!86 = distinct !{!86, !53}
!87 = distinct !{!87, !53}
!88 = distinct !{!88, !53}
!89 = distinct !{!89, !53}
!90 = distinct !{!90, !53}
!91 = distinct !{!91, !53}
!92 = distinct !{!92, !53}
!93 = distinct !{!93, !53}
