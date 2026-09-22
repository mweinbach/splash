; ModuleID = '/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/_cpu_build_v1/candidate.air'
source_filename = "/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/candidate.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v29-apple-macosx27.0.0"

%"struct.metal::_atomic" = type { i32 }
%struct.FlashAffineParams = type { i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64 }

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_timed_q4_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11, i32 noundef %12) local_unnamed_addr #0 {
  %14 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %15 = tail call i1 @air.any.v3i1(<3 x i1> %14) #9
  %16 = icmp ne i32 %10, 32
  %17 = or i1 %16, %15
  %18 = icmp ugt i32 %11, 1
  %19 = or i1 %18, %17
  %20 = icmp ugt i32 %12, 31
  %21 = or i1 %20, %19
  br i1 %21, label %22, label %27

22:                                               ; preds = %13
  %23 = icmp eq i32 %12, 0
  br i1 %23, label %24, label %28

24:                                               ; preds = %22
  %25 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %26 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %25, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %28

27:                                               ; preds = %13
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt4ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %11, i32 noundef %12) #11
  br label %28

28:                                               ; preds = %27, %24, %22
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
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !37
  switch i32 %18, label %84 [
    i32 2560, label %27
    i32 6144, label %19
  ]

19:                                               ; preds = %10
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !43
  %22 = icmp eq i32 %21, 2560
  %23 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %24 = load i32, i32 addrspace(2)* %23, align 8
  %25 = icmp eq i32 %24, 4
  %26 = select i1 %22, i1 %25, i1 false
  br i1 %26, label %34, label %84

27:                                               ; preds = %10
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %29 = load i32, i32 addrspace(2)* %28, align 4, !tbaa !43
  switch i32 %29, label %84 [
    i32 10240, label %30
    i32 12288, label %30
  ]

30:                                               ; preds = %27, %27
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %32 = load i32, i32 addrspace(2)* %31, align 8, !tbaa !44
  %33 = icmp eq i32 %32, 4
  br i1 %33, label %34, label %84

34:                                               ; preds = %30, %19
  %35 = phi i32 [ 2560, %19 ], [ %29, %30 ]
  %36 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %37 = load i32, i32 addrspace(2)* %36, align 4, !tbaa !45
  %38 = icmp eq i32 %37, 1
  br i1 %38, label %39, label %84

39:                                               ; preds = %34
  %40 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %41 = load i32, i32 addrspace(2)* %40, align 8, !tbaa !46
  %42 = icmp eq i32 %41, 1
  br i1 %42, label %43, label %84

43:                                               ; preds = %39
  %44 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %45 = load i32, i32 addrspace(2)* %44, align 4, !tbaa !47
  %46 = icmp eq i32 %45, 0
  br i1 %46, label %47, label %84

47:                                               ; preds = %43
  %48 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %49 = load i32, i32 addrspace(2)* %48, align 4, !tbaa !48
  %50 = icmp eq i32 %49, 4
  br i1 %50, label %51, label %84

51:                                               ; preds = %47
  %52 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %53 = load i32, i32 addrspace(2)* %52, align 8, !tbaa !49
  %54 = icmp eq i32 %53, 64
  %55 = and i32 %18, 511
  %56 = icmp eq i32 %55, 0
  %57 = select i1 %54, i1 %56, i1 false
  %58 = and i32 %35, 7
  %59 = icmp eq i32 %58, 0
  %60 = select i1 %57, i1 %59, i1 false
  br i1 %60, label %61, label %84

61:                                               ; preds = %51
  %62 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %63 = load i64, i64 addrspace(2)* %62, align 8, !tbaa !50
  %64 = lshr i32 %18, 1
  %65 = zext i32 %64 to i64
  %66 = icmp ult i64 %63, %65
  br i1 %66, label %84, label %67

67:                                               ; preds = %61
  %68 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %69 = load i64, i64 addrspace(2)* %68, align 8, !tbaa !51
  %70 = lshr i32 %18, 5
  %71 = and i32 %70, 134217726
  %72 = zext i32 %71 to i64
  %73 = icmp uge i64 %69, %72
  %74 = and i64 %69, 1
  %75 = icmp eq i64 %74, 0
  %76 = and i1 %73, %75
  br i1 %76, label %77, label %84

77:                                               ; preds = %67
  %78 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %79 = load i64, i64 addrspace(2)* %78, align 8, !tbaa !52
  %80 = and i64 %79, 1
  %81 = icmp eq i64 %80, 0
  br i1 %81, label %82, label %84

82:                                               ; preds = %77
  %83 = tail call zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt4ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6) #12
  br i1 %83, label %89, label %84

84:                                               ; preds = %82, %77, %67, %61, %51, %47, %43, %39, %34, %30, %27, %19, %10
  %85 = icmp eq i32 %9, 0
  br i1 %85, label %86, label %219

86:                                               ; preds = %84
  %87 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %88 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %87, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %219

89:                                               ; preds = %82
  %90 = extractelement <3 x i32> %7, i64 0
  %91 = shl i32 %90, 3
  %92 = shl i32 %8, 2
  %93 = add i32 %91, %92
  %94 = lshr i32 %35, 3
  %95 = icmp uge i32 %90, %94
  %96 = extractelement <3 x i32> %7, i64 1
  %97 = icmp ugt i32 %96, 1
  %98 = or i1 %97, %95
  %99 = extractelement <3 x i32> %7, i64 2
  %100 = icmp ne i32 %99, 0
  %101 = or i1 %100, %98
  %102 = xor i1 %101, true
  %103 = icmp ult i32 %93, %35
  %104 = select i1 %102, i1 %103, i1 false
  br i1 %104, label %105, label %219

105:                                              ; preds = %89
  %106 = zext i32 %96 to i64
  %107 = shl nuw nsw i64 %106, 1
  %108 = or i64 %107, 1
  %109 = zext i32 %18 to i64
  %110 = mul nuw nsw i64 %107, %109
  %111 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %110
  %112 = mul i64 %108, %109
  %113 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %112
  %114 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %114) #13
  %115 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %115) #13
  %116 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %116) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %116, i8 0, i64 16, i1 false)
  %117 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %117) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %117, i8 0, i64 16, i1 false)
  %118 = shl i32 %9, 4
  %119 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 0
  %120 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %121 = bitcast float* %15 to i8*
  %122 = bitcast float* %16 to i8*
  br label %132

123:                                              ; preds = %144
  %124 = icmp eq i32 %9, 0
  %125 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %126 = zext i32 %35 to i64
  %127 = mul i64 %107, %126
  %128 = zext i32 %93 to i64
  %129 = add i64 %127, %128
  %130 = mul i64 %108, %126
  %131 = add i64 %130, %128
  br label %176

132:                                              ; preds = %144, %105
  %133 = phi i32 [ 0, %105 ], [ %145, %144 ]
  %134 = add i32 %133, %118
  %135 = zext i32 %134 to i64
  %136 = getelementptr inbounds bfloat, bfloat addrspace(1)* %111, i64 %135
  %137 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %136, float* noundef nonnull %119) #12
  %138 = getelementptr inbounds bfloat, bfloat addrspace(1)* %113, i64 %135
  %139 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %138, float* noundef nonnull %120) #12
  %140 = lshr i32 %134, 6
  %141 = lshr exact i64 %135, 1
  %142 = zext i32 %140 to i64
  %143 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %141
  br label %147

144:                                              ; preds = %147
  %145 = add i32 %133, 512
  %146 = icmp ult i32 %145, %18
  br i1 %146, label %132, label %123, !llvm.loop !53

147:                                              ; preds = %147, %132
  %148 = phi i32 [ 0, %132 ], [ %173, %147 ]
  %149 = add nuw nsw i32 %148, %93
  %150 = zext i32 %149 to i64
  %151 = mul i64 %63, %150
  %152 = getelementptr inbounds i8, i8 addrspace(1)* %143, i64 %151
  %153 = mul i64 %69, %150
  %154 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %153
  %155 = bitcast i8 addrspace(1)* %154 to bfloat addrspace(1)*
  %156 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %153
  %157 = bitcast i8 addrspace(1)* %156 to bfloat addrspace(1)*
  %158 = getelementptr inbounds bfloat, bfloat addrspace(1)* %155, i64 %142
  %159 = load bfloat, bfloat addrspace(1)* %158, align 2, !tbaa !55
  %160 = fpext bfloat %159 to float
  %161 = getelementptr inbounds bfloat, bfloat addrspace(1)* %157, i64 %142
  %162 = load bfloat, bfloat addrspace(1)* %161, align 2, !tbaa !55
  %163 = fpext bfloat %162 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %121) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %122) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %152, float* noundef nonnull %119, float* noundef nonnull %120, float noundef %160, float noundef %163, float noundef %137, float noundef %139, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #12
  %164 = load float, float* %15, align 4, !tbaa !57
  %165 = zext i32 %148 to i64
  %166 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %165
  %167 = load float, float* %166, align 4, !tbaa !57
  %168 = fadd float %164, %167
  store float %168, float* %166, align 4, !tbaa !57
  %169 = load float, float* %16, align 4, !tbaa !57
  %170 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %165
  %171 = load float, float* %170, align 4, !tbaa !57
  %172 = fadd float %169, %171
  store float %172, float* %170, align 4, !tbaa !57
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %122) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %121) #13
  %173 = add nuw nsw i32 %148, 1
  %174 = icmp eq i32 %173, 4
  br i1 %174, label %144, label %147, !llvm.loop !59

175:                                              ; preds = %216
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %117) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %116) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %115) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %114) #13
  br label %219

176:                                              ; preds = %216, %123
  %177 = phi i16 [ 0, %123 ], [ %217, %216 ]
  %178 = zext i16 %177 to i64
  %179 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %178
  %180 = load float, float* %179, align 4, !tbaa !57
  %181 = call fast float @air.simd_sum.f32(float %180) #14
  br i1 %124, label %182, label %197

182:                                              ; preds = %176
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
  %193 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %125, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %194

194:                                              ; preds = %192, %187
  %195 = add i64 %129, %178
  %196 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %195
  store bfloat %183, bfloat addrspace(1)* %196, align 2, !tbaa !55
  br label %197

197:                                              ; preds = %194, %176
  %198 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %178
  %199 = load float, float* %198, align 4, !tbaa !57
  %200 = call fast float @air.simd_sum.f32(float %199) #14
  br i1 %124, label %201, label %216

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
  %212 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %125, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %213

213:                                              ; preds = %211, %206
  %214 = add i64 %131, %178
  %215 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %214
  store bfloat %202, bfloat addrspace(1)* %215, align 2, !tbaa !55
  br label %216

216:                                              ; preds = %213, %197
  %217 = add nuw nsw i16 %177, 1
  %218 = icmp eq i16 %217, 4
  br i1 %218, label %175, label %176, !llvm.loop !60

219:                                              ; preds = %175, %89, %86, %84
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_timed_q5_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11, i32 noundef %12) local_unnamed_addr #0 {
  %14 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %15 = tail call i1 @air.any.v3i1(<3 x i1> %14) #9
  %16 = icmp ne i32 %10, 32
  %17 = or i1 %16, %15
  %18 = icmp ugt i32 %11, 1
  %19 = or i1 %18, %17
  %20 = icmp ugt i32 %12, 31
  %21 = or i1 %20, %19
  br i1 %21, label %22, label %27

22:                                               ; preds = %13
  %23 = icmp eq i32 %12, 0
  br i1 %23, label %24, label %28

24:                                               ; preds = %22
  %25 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %26 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %25, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %28

27:                                               ; preds = %13
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt5ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %11, i32 noundef %12) #11
  br label %28

28:                                               ; preds = %27, %24, %22
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
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !37
  %19 = icmp eq i32 %18, 2560
  br i1 %19, label %20, label %66

20:                                               ; preds = %10
  %21 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %22 = load i32, i32 addrspace(2)* %21, align 4, !tbaa !43
  %23 = icmp eq i32 %22, 10240
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %25 = load i32, i32 addrspace(2)* %24, align 8
  %26 = icmp eq i32 %25, 4
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %28, label %66

28:                                               ; preds = %20
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !45
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %66

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %34 = load i32, i32 addrspace(2)* %33, align 8, !tbaa !46
  %35 = icmp eq i32 %34, 1
  br i1 %35, label %36, label %66

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !47
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %66

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %42 = load i32, i32 addrspace(2)* %41, align 4, !tbaa !48
  %43 = icmp eq i32 %42, 5
  br i1 %43, label %44, label %66

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %46 = load i32, i32 addrspace(2)* %45, align 8, !tbaa !49
  %47 = icmp eq i32 %46, 64
  br i1 %47, label %48, label %66

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %50 = load i64, i64 addrspace(2)* %49, align 8, !tbaa !50
  %51 = icmp ult i64 %50, 1600
  br i1 %51, label %66, label %52

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %54 = load i64, i64 addrspace(2)* %53, align 8, !tbaa !51
  %55 = icmp ugt i64 %54, 79
  %56 = and i64 %54, 1
  %57 = icmp eq i64 %56, 0
  %58 = and i1 %55, %57
  br i1 %58, label %59, label %66

59:                                               ; preds = %52
  %60 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %61 = load i64, i64 addrspace(2)* %60, align 8, !tbaa !52
  %62 = and i64 %61, 1
  %63 = icmp eq i64 %62, 0
  br i1 %63, label %64, label %66

64:                                               ; preds = %59
  %65 = tail call zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt5ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6) #12
  br i1 %65, label %71, label %66

66:                                               ; preds = %64, %59, %52, %48, %44, %40, %36, %32, %28, %20, %10
  %67 = icmp eq i32 %9, 0
  br i1 %67, label %68, label %198

68:                                               ; preds = %66
  %69 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %70 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %69, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %198

71:                                               ; preds = %64
  %72 = extractelement <3 x i32> %7, i64 0
  %73 = shl i32 %72, 3
  %74 = shl i32 %8, 2
  %75 = add i32 %73, %74
  %76 = icmp ugt i32 %72, 1279
  %77 = extractelement <3 x i32> %7, i64 1
  %78 = icmp ugt i32 %77, 1
  %79 = or i1 %78, %76
  %80 = extractelement <3 x i32> %7, i64 2
  %81 = icmp ne i32 %80, 0
  %82 = or i1 %81, %79
  %83 = icmp ugt i32 %75, 10239
  %84 = or i1 %82, %83
  br i1 %84, label %198, label %85

85:                                               ; preds = %71
  %86 = zext i32 %77 to i64
  %87 = shl nuw nsw i64 %86, 1
  %88 = or i64 %87, 1
  %89 = mul nuw nsw i64 %86, 5120
  %90 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %89
  %91 = mul nuw nsw i64 %88, 2560
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %91
  %93 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %93) #13
  %94 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %94) #13
  %95 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %95) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %95, i8 0, i64 16, i1 false)
  %96 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %96, i8 0, i64 16, i1 false)
  %97 = shl i32 %9, 4
  %98 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 0
  %99 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %100 = bitcast float* %15 to i8*
  %101 = bitcast float* %16 to i8*
  br label %110

102:                                              ; preds = %123
  %103 = icmp eq i32 %9, 0
  %104 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %105 = mul nuw nsw i64 %86, 20480
  %106 = zext i32 %75 to i64
  %107 = add nuw nsw i64 %105, %106
  %108 = mul nuw nsw i64 %88, 10240
  %109 = add nuw nsw i64 %108, %106
  br label %155

110:                                              ; preds = %123, %85
  %111 = phi i32 [ 0, %85 ], [ %124, %123 ]
  %112 = add i32 %111, %97
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds bfloat, bfloat addrspace(1)* %90, i64 %113
  %115 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %114, float* noundef nonnull %98) #12
  %116 = getelementptr inbounds bfloat, bfloat addrspace(1)* %92, i64 %113
  %117 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %116, float* noundef nonnull %99) #12
  %118 = lshr i32 %112, 6
  %119 = mul nuw nsw i64 %113, 5
  %120 = lshr exact i64 %119, 3
  %121 = zext i32 %118 to i64
  %122 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %120
  br label %126

123:                                              ; preds = %126
  %124 = add i32 %111, 512
  %125 = icmp ult i32 %124, 2560
  br i1 %125, label %110, label %102, !llvm.loop !61

126:                                              ; preds = %126, %110
  %127 = phi i32 [ 0, %110 ], [ %152, %126 ]
  %128 = add nuw nsw i32 %127, %75
  %129 = zext i32 %128 to i64
  %130 = mul i64 %50, %129
  %131 = getelementptr inbounds i8, i8 addrspace(1)* %122, i64 %130
  %132 = mul i64 %54, %129
  %133 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %132
  %134 = bitcast i8 addrspace(1)* %133 to bfloat addrspace(1)*
  %135 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %132
  %136 = bitcast i8 addrspace(1)* %135 to bfloat addrspace(1)*
  %137 = getelementptr inbounds bfloat, bfloat addrspace(1)* %134, i64 %121
  %138 = load bfloat, bfloat addrspace(1)* %137, align 2, !tbaa !55
  %139 = fpext bfloat %138 to float
  %140 = getelementptr inbounds bfloat, bfloat addrspace(1)* %136, i64 %121
  %141 = load bfloat, bfloat addrspace(1)* %140, align 2, !tbaa !55
  %142 = fpext bfloat %141 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %100) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %101) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %131, float* noundef nonnull %98, float* noundef nonnull %99, float noundef %139, float noundef %142, float noundef %115, float noundef %117, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #12
  %143 = load float, float* %15, align 4, !tbaa !57
  %144 = zext i32 %127 to i64
  %145 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %144
  %146 = load float, float* %145, align 4, !tbaa !57
  %147 = fadd float %143, %146
  store float %147, float* %145, align 4, !tbaa !57
  %148 = load float, float* %16, align 4, !tbaa !57
  %149 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %144
  %150 = load float, float* %149, align 4, !tbaa !57
  %151 = fadd float %148, %150
  store float %151, float* %149, align 4, !tbaa !57
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %101) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %100) #13
  %152 = add nuw nsw i32 %127, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %123, label %126, !llvm.loop !62

154:                                              ; preds = %195
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %95) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %94) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %93) #13
  br label %198

155:                                              ; preds = %195, %102
  %156 = phi i16 [ 0, %102 ], [ %196, %195 ]
  %157 = zext i16 %156 to i64
  %158 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !57
  %160 = call fast float @air.simd_sum.f32(float %159) #14
  br i1 %103, label %161, label %176

161:                                              ; preds = %155
  %162 = fptrunc float %160 to bfloat
  %163 = bitcast float %160 to i32
  %164 = and i32 %163, 2139095040
  %165 = icmp eq i32 %164, 2139095040
  br i1 %165, label %171, label %166

166:                                              ; preds = %161
  %167 = fpext bfloat %162 to float
  %168 = bitcast float %167 to i32
  %169 = and i32 %168, 2139095040
  %170 = icmp eq i32 %169, 2139095040
  br i1 %170, label %171, label %173

171:                                              ; preds = %166, %161
  %172 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %104, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %173

173:                                              ; preds = %171, %166
  %174 = add nuw nsw i64 %107, %157
  %175 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %174
  store bfloat %162, bfloat addrspace(1)* %175, align 2, !tbaa !55
  br label %176

176:                                              ; preds = %173, %155
  %177 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %157
  %178 = load float, float* %177, align 4, !tbaa !57
  %179 = call fast float @air.simd_sum.f32(float %178) #14
  br i1 %103, label %180, label %195

180:                                              ; preds = %176
  %181 = fptrunc float %179 to bfloat
  %182 = bitcast float %179 to i32
  %183 = and i32 %182, 2139095040
  %184 = icmp eq i32 %183, 2139095040
  br i1 %184, label %190, label %185

185:                                              ; preds = %180
  %186 = fpext bfloat %181 to float
  %187 = bitcast float %186 to i32
  %188 = and i32 %187, 2139095040
  %189 = icmp eq i32 %188, 2139095040
  br i1 %189, label %190, label %192

190:                                              ; preds = %185, %180
  %191 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %104, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %192

192:                                              ; preds = %190, %185
  %193 = add nuw nsw i64 %109, %157
  %194 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %193
  store bfloat %181, bfloat addrspace(1)* %194, align 2, !tbaa !55
  br label %195

195:                                              ; preds = %192, %176
  %196 = add nuw nsw i16 %156, 1
  %197 = icmp eq i16 %196, 4
  br i1 %197, label %154, label %155, !llvm.loop !63

198:                                              ; preds = %154, %71, %68, %66
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_timed_q5_g128(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11, i32 noundef %12) local_unnamed_addr #0 {
  %14 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %15 = tail call i1 @air.any.v3i1(<3 x i1> %14) #9
  %16 = icmp ne i32 %10, 32
  %17 = or i1 %16, %15
  %18 = icmp ugt i32 %11, 1
  %19 = or i1 %18, %17
  %20 = icmp ugt i32 %12, 31
  %21 = or i1 %20, %19
  br i1 %21, label %22, label %27

22:                                               ; preds = %13
  %23 = icmp eq i32 %12, 0
  br i1 %23, label %24, label %28

24:                                               ; preds = %22
  %25 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %26 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %25, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %28

27:                                               ; preds = %13
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt5ELt128EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %11, i32 noundef %12) #11
  br label %28

28:                                               ; preds = %27, %24, %22
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
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !37
  switch i32 %18, label %83 [
    i32 6144, label %27
    i32 2560, label %19
  ]

19:                                               ; preds = %10
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4, !tbaa !43
  %22 = icmp eq i32 %21, 6144
  %23 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %24 = load i32, i32 addrspace(2)* %23, align 8
  %25 = icmp eq i32 %24, 4
  %26 = select i1 %22, i1 %25, i1 false
  br i1 %26, label %35, label %83

27:                                               ; preds = %10
  %28 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %29 = load i32, i32 addrspace(2)* %28, align 4, !tbaa !43
  %30 = icmp eq i32 %29, 2560
  %31 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %32 = load i32, i32 addrspace(2)* %31, align 8
  %33 = icmp eq i32 %32, 4
  %34 = select i1 %30, i1 %33, i1 false
  br i1 %34, label %35, label %83

35:                                               ; preds = %27, %19
  %36 = phi i32 [ 6144, %19 ], [ 2560, %27 ]
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !45
  %39 = icmp eq i32 %38, 1
  br i1 %39, label %40, label %83

40:                                               ; preds = %35
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %42 = load i32, i32 addrspace(2)* %41, align 8, !tbaa !46
  %43 = icmp eq i32 %42, 1
  br i1 %43, label %44, label %83

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %46 = load i32, i32 addrspace(2)* %45, align 4, !tbaa !47
  %47 = icmp eq i32 %46, 0
  br i1 %47, label %48, label %83

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %50 = load i32, i32 addrspace(2)* %49, align 4, !tbaa !48
  %51 = icmp eq i32 %50, 5
  br i1 %51, label %52, label %83

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %54 = load i32, i32 addrspace(2)* %53, align 8, !tbaa !49
  %55 = icmp eq i32 %54, 128
  %56 = and i32 %18, 511
  %57 = icmp eq i32 %56, 0
  %58 = select i1 %55, i1 %57, i1 false
  br i1 %58, label %59, label %83

59:                                               ; preds = %52
  %60 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %61 = load i64, i64 addrspace(2)* %60, align 8, !tbaa !50
  %62 = zext i32 %18 to i64
  %63 = mul nuw nsw i64 %62, 5
  %64 = lshr i64 %63, 3
  %65 = icmp ult i64 %61, %64
  br i1 %65, label %83, label %66

66:                                               ; preds = %59
  %67 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %68 = load i64, i64 addrspace(2)* %67, align 8, !tbaa !51
  %69 = lshr i32 %18, 6
  %70 = and i32 %69, 67108862
  %71 = zext i32 %70 to i64
  %72 = icmp uge i64 %68, %71
  %73 = and i64 %68, 1
  %74 = icmp eq i64 %73, 0
  %75 = and i1 %72, %74
  br i1 %75, label %76, label %83

76:                                               ; preds = %66
  %77 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %78 = load i64, i64 addrspace(2)* %77, align 8, !tbaa !52
  %79 = and i64 %78, 1
  %80 = icmp eq i64 %79, 0
  br i1 %80, label %81, label %83

81:                                               ; preds = %76
  %82 = tail call zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt5ELt128EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6) #12
  br i1 %82, label %88, label %83

83:                                               ; preds = %81, %76, %66, %59, %52, %48, %44, %40, %35, %27, %19, %10
  %84 = icmp eq i32 %9, 0
  br i1 %84, label %85, label %217

85:                                               ; preds = %83
  %86 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %87 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %86, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %217

88:                                               ; preds = %81
  %89 = extractelement <3 x i32> %7, i64 0
  %90 = shl i32 %89, 3
  %91 = shl i32 %8, 2
  %92 = add i32 %90, %91
  %93 = lshr exact i32 %36, 3
  %94 = icmp uge i32 %89, %93
  %95 = extractelement <3 x i32> %7, i64 1
  %96 = icmp ugt i32 %95, 1
  %97 = or i1 %96, %94
  %98 = extractelement <3 x i32> %7, i64 2
  %99 = icmp ne i32 %98, 0
  %100 = or i1 %99, %97
  %101 = icmp uge i32 %92, %36
  %102 = or i1 %100, %101
  br i1 %102, label %217, label %103

103:                                              ; preds = %88
  %104 = zext i32 %95 to i64
  %105 = shl nuw nsw i64 %104, 1
  %106 = or i64 %105, 1
  %107 = mul nuw nsw i64 %105, %62
  %108 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %107
  %109 = mul i64 %106, %62
  %110 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %109
  %111 = bitcast [16 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %111) #13
  %112 = bitcast [16 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* nonnull %112) #13
  %113 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %113) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %113, i8 0, i64 16, i1 false)
  %114 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %114) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %114, i8 0, i64 16, i1 false)
  %115 = shl i32 %9, 4
  %116 = getelementptr inbounds [16 x float], [16 x float]* %11, i64 0, i64 0
  %117 = getelementptr inbounds [16 x float], [16 x float]* %12, i64 0, i64 0
  %118 = bitcast float* %15 to i8*
  %119 = bitcast float* %16 to i8*
  br label %129

120:                                              ; preds = %142
  %121 = icmp eq i32 %9, 0
  %122 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %123 = zext i32 %36 to i64
  %124 = mul nuw nsw i64 %105, %123
  %125 = zext i32 %92 to i64
  %126 = add nuw nsw i64 %124, %125
  %127 = mul nuw nsw i64 %106, %123
  %128 = add nuw nsw i64 %127, %125
  br label %174

129:                                              ; preds = %142, %103
  %130 = phi i32 [ 0, %103 ], [ %143, %142 ]
  %131 = add i32 %130, %115
  %132 = zext i32 %131 to i64
  %133 = getelementptr inbounds bfloat, bfloat addrspace(1)* %108, i64 %132
  %134 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %133, float* noundef nonnull %116) #12
  %135 = getelementptr inbounds bfloat, bfloat addrspace(1)* %110, i64 %132
  %136 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %135, float* noundef nonnull %117) #12
  %137 = lshr i32 %131, 7
  %138 = mul nuw nsw i64 %132, 5
  %139 = lshr exact i64 %138, 3
  %140 = zext i32 %137 to i64
  %141 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %139
  br label %145

142:                                              ; preds = %145
  %143 = add i32 %130, 512
  %144 = icmp ult i32 %143, %18
  br i1 %144, label %129, label %120, !llvm.loop !64

145:                                              ; preds = %145, %129
  %146 = phi i32 [ 0, %129 ], [ %171, %145 ]
  %147 = add nuw nsw i32 %146, %92
  %148 = zext i32 %147 to i64
  %149 = mul i64 %61, %148
  %150 = getelementptr inbounds i8, i8 addrspace(1)* %141, i64 %149
  %151 = mul i64 %68, %148
  %152 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %151
  %153 = bitcast i8 addrspace(1)* %152 to bfloat addrspace(1)*
  %154 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %151
  %155 = bitcast i8 addrspace(1)* %154 to bfloat addrspace(1)*
  %156 = getelementptr inbounds bfloat, bfloat addrspace(1)* %153, i64 %140
  %157 = load bfloat, bfloat addrspace(1)* %156, align 2, !tbaa !55
  %158 = fpext bfloat %157 to float
  %159 = getelementptr inbounds bfloat, bfloat addrspace(1)* %155, i64 %140
  %160 = load bfloat, bfloat addrspace(1)* %159, align 2, !tbaa !55
  %161 = fpext bfloat %160 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %118) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %119) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %150, float* noundef nonnull %116, float* noundef nonnull %117, float noundef %158, float noundef %161, float noundef %134, float noundef %136, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #12
  %162 = load float, float* %15, align 4, !tbaa !57
  %163 = zext i32 %146 to i64
  %164 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %163
  %165 = load float, float* %164, align 4, !tbaa !57
  %166 = fadd float %162, %165
  store float %166, float* %164, align 4, !tbaa !57
  %167 = load float, float* %16, align 4, !tbaa !57
  %168 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %163
  %169 = load float, float* %168, align 4, !tbaa !57
  %170 = fadd float %167, %169
  store float %170, float* %168, align 4, !tbaa !57
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %119) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %118) #13
  %171 = add nuw nsw i32 %146, 1
  %172 = icmp eq i32 %171, 4
  br i1 %172, label %142, label %145, !llvm.loop !65

173:                                              ; preds = %214
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %114) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %113) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %112) #13
  call void @llvm.lifetime.end.p0i8(i64 64, i8* nonnull %111) #13
  br label %217

174:                                              ; preds = %214, %120
  %175 = phi i16 [ 0, %120 ], [ %215, %214 ]
  %176 = zext i16 %175 to i64
  %177 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %176
  %178 = load float, float* %177, align 4, !tbaa !57
  %179 = call fast float @air.simd_sum.f32(float %178) #14
  br i1 %121, label %180, label %195

180:                                              ; preds = %174
  %181 = fptrunc float %179 to bfloat
  %182 = bitcast float %179 to i32
  %183 = and i32 %182, 2139095040
  %184 = icmp eq i32 %183, 2139095040
  br i1 %184, label %190, label %185

185:                                              ; preds = %180
  %186 = fpext bfloat %181 to float
  %187 = bitcast float %186 to i32
  %188 = and i32 %187, 2139095040
  %189 = icmp eq i32 %188, 2139095040
  br i1 %189, label %190, label %192

190:                                              ; preds = %185, %180
  %191 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %122, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %192

192:                                              ; preds = %190, %185
  %193 = add nuw nsw i64 %126, %176
  %194 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %193
  store bfloat %181, bfloat addrspace(1)* %194, align 2, !tbaa !55
  br label %195

195:                                              ; preds = %192, %174
  %196 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %176
  %197 = load float, float* %196, align 4, !tbaa !57
  %198 = call fast float @air.simd_sum.f32(float %197) #14
  br i1 %121, label %199, label %214

199:                                              ; preds = %195
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
  %210 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %122, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %211

211:                                              ; preds = %209, %204
  %212 = add nuw nsw i64 %128, %176
  %213 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %212
  store bfloat %200, bfloat addrspace(1)* %213, align 2, !tbaa !55
  br label %214

214:                                              ; preds = %211, %195
  %215 = add nuw nsw i16 %175, 1
  %216 = icmp eq i16 %215, 4
  br i1 %216, label %173, label %174, !llvm.loop !66

217:                                              ; preds = %173, %88, %85, %83
  ret void
}

; Function Attrs: convergent mustprogress nounwind
define void @raw_large_R4_guard_pair_sep22_timed_q6_g64(bfloat addrspace(1)* noundef "air-buffer-no-alias" %0, i8 addrspace(1)* noundef "air-buffer-no-alias" %1, i8 addrspace(1)* noundef "air-buffer-no-alias" %2, i8 addrspace(1)* noundef "air-buffer-no-alias" %3, i64 addrspace(1)* nocapture noundef readnone "air-buffer-no-alias" %4, bfloat addrspace(1)* noundef "air-buffer-no-alias" %5, %"struct.metal::_atomic" addrspace(1)* noundef "air-buffer-no-alias" %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) "air-buffer-no-alias" %7, <3 x i32> noundef %8, <3 x i32> noundef %9, i32 noundef %10, i32 noundef %11, i32 noundef %12) local_unnamed_addr #0 {
  %14 = icmp ne <3 x i32> %9, <i32 64, i32 1, i32 1>
  %15 = tail call i1 @air.any.v3i1(<3 x i1> %14) #9
  %16 = icmp ne i32 %10, 32
  %17 = or i1 %16, %15
  %18 = icmp ugt i32 %11, 1
  %19 = or i1 %18, %17
  %20 = icmp ugt i32 %12, 31
  %21 = or i1 %20, %19
  br i1 %21, label %22, label %27

22:                                               ; preds = %13
  %23 = icmp eq i32 %12, 0
  br i1 %23, label %24, label %28

24:                                               ; preds = %22
  %25 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %6, i64 0, i32 0
  %26 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %25, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %28

27:                                               ; preds = %13
  tail call void @_ZN24r5_raw_odd_rowpair_sep2212project_pairILt6ELt64EEEvPU9MTLdeviceKDF16bPU9MTLdeviceKhS4_S4_PU9MTLdeviceDF16bPU9MTLdeviceN5metal7_atomicIjLNS7_12thread_scopeE2EvEERU11MTLconstantK17FlashAffineParamsDv3_jjj(bfloat addrspace(1)* noundef %0, i8 addrspace(1)* noundef %1, i8 addrspace(1)* noundef %2, i8 addrspace(1)* noundef %3, bfloat addrspace(1)* noundef %5, %"struct.metal::_atomic" addrspace(1)* noundef %6, %struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %7, <3 x i32> noundef %8, i32 noundef %11, i32 noundef %12) #11
  br label %28

28:                                               ; preds = %27, %24, %22
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
  %18 = load i32, i32 addrspace(2)* %17, align 8, !tbaa !37
  %19 = icmp eq i32 %18, 2560
  %20 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 3
  %21 = load i32, i32 addrspace(2)* %20, align 4
  %22 = icmp eq i32 %21, 6144
  %23 = select i1 %19, i1 %22, i1 false
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 0
  %25 = load i32, i32 addrspace(2)* %24, align 8
  %26 = icmp eq i32 %25, 4
  %27 = select i1 %23, i1 %26, i1 false
  br i1 %27, label %28, label %66

28:                                               ; preds = %10
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 1
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !45
  %31 = icmp eq i32 %30, 1
  br i1 %31, label %32, label %66

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 4
  %34 = load i32, i32 addrspace(2)* %33, align 8, !tbaa !46
  %35 = icmp eq i32 %34, 1
  br i1 %35, label %36, label %66

36:                                               ; preds = %32
  %37 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 7
  %38 = load i32, i32 addrspace(2)* %37, align 4, !tbaa !47
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %66

40:                                               ; preds = %36
  %41 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 5
  %42 = load i32, i32 addrspace(2)* %41, align 4, !tbaa !48
  %43 = icmp eq i32 %42, 6
  br i1 %43, label %44, label %66

44:                                               ; preds = %40
  %45 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 6
  %46 = load i32, i32 addrspace(2)* %45, align 8, !tbaa !49
  %47 = icmp eq i32 %46, 64
  br i1 %47, label %48, label %66

48:                                               ; preds = %44
  %49 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 8
  %50 = load i64, i64 addrspace(2)* %49, align 8, !tbaa !50
  %51 = icmp ult i64 %50, 1920
  br i1 %51, label %66, label %52

52:                                               ; preds = %48
  %53 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 10
  %54 = load i64, i64 addrspace(2)* %53, align 8, !tbaa !51
  %55 = icmp ugt i64 %54, 79
  %56 = and i64 %54, 1
  %57 = icmp eq i64 %56, 0
  %58 = and i1 %55, %57
  br i1 %58, label %59, label %66

59:                                               ; preds = %52
  %60 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %6, i64 0, i32 11
  %61 = load i64, i64 addrspace(2)* %60, align 8, !tbaa !52
  %62 = and i64 %61, 1
  %63 = icmp eq i64 %62, 0
  br i1 %63, label %64, label %66

64:                                               ; preds = %59
  %65 = tail call zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt6ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %6) #12
  br i1 %65, label %71, label %66

66:                                               ; preds = %64, %59, %52, %48, %44, %40, %36, %32, %28, %10
  %67 = icmp eq i32 %9, 0
  br i1 %67, label %68, label %198

68:                                               ; preds = %66
  %69 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %70 = tail call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %69, i32 2, i32 0, i32 2, i32 0, i1 false) #10
  br label %198

71:                                               ; preds = %64
  %72 = extractelement <3 x i32> %7, i64 0
  %73 = shl i32 %72, 3
  %74 = shl i32 %8, 2
  %75 = add i32 %73, %74
  %76 = icmp ugt i32 %72, 767
  %77 = extractelement <3 x i32> %7, i64 1
  %78 = icmp ugt i32 %77, 1
  %79 = or i1 %78, %76
  %80 = extractelement <3 x i32> %7, i64 2
  %81 = icmp ne i32 %80, 0
  %82 = or i1 %81, %79
  %83 = icmp ugt i32 %75, 6143
  %84 = or i1 %82, %83
  br i1 %84, label %198, label %85

85:                                               ; preds = %71
  %86 = zext i32 %77 to i64
  %87 = shl nuw nsw i64 %86, 1
  %88 = or i64 %87, 1
  %89 = mul nuw nsw i64 %86, 5120
  %90 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %89
  %91 = mul nuw nsw i64 %88, 2560
  %92 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %91
  %93 = bitcast [8 x float]* %11 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %93) #13
  %94 = bitcast [8 x float]* %12 to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %94) #13
  %95 = bitcast [4 x float]* %13 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %95) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %95, i8 0, i64 16, i1 false)
  %96 = bitcast [4 x float]* %14 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.memset.p0i8.i64(i8* noundef nonnull align 4 dereferenceable(16) %96, i8 0, i64 16, i1 false)
  %97 = shl i32 %9, 3
  %98 = getelementptr inbounds [8 x float], [8 x float]* %11, i64 0, i64 0
  %99 = getelementptr inbounds [8 x float], [8 x float]* %12, i64 0, i64 0
  %100 = bitcast float* %15 to i8*
  %101 = bitcast float* %16 to i8*
  br label %110

102:                                              ; preds = %123
  %103 = icmp eq i32 %9, 0
  %104 = getelementptr inbounds %"struct.metal::_atomic", %"struct.metal::_atomic" addrspace(1)* %5, i64 0, i32 0
  %105 = mul nuw nsw i64 %86, 12288
  %106 = zext i32 %75 to i64
  %107 = add nuw nsw i64 %105, %106
  %108 = mul nuw nsw i64 %88, 6144
  %109 = add nuw nsw i64 %108, %106
  br label %155

110:                                              ; preds = %123, %85
  %111 = phi i32 [ 0, %85 ], [ %124, %123 ]
  %112 = add i32 %111, %97
  %113 = zext i32 %112 to i64
  %114 = getelementptr inbounds bfloat, bfloat addrspace(1)* %90, i64 %113
  %115 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %114, float* noundef nonnull %98) #12
  %116 = getelementptr inbounds bfloat, bfloat addrspace(1)* %92, i64 %113
  %117 = call fast float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %116, float* noundef nonnull %99) #12
  %118 = lshr i32 %112, 6
  %119 = mul nuw nsw i64 %113, 6
  %120 = lshr exact i64 %119, 3
  %121 = zext i32 %118 to i64
  %122 = getelementptr inbounds i8, i8 addrspace(1)* %1, i64 %120
  br label %126

123:                                              ; preds = %126
  %124 = add i32 %111, 256
  %125 = icmp ult i32 %124, 2560
  br i1 %125, label %110, label %102, !llvm.loop !67

126:                                              ; preds = %126, %110
  %127 = phi i32 [ 0, %110 ], [ %152, %126 ]
  %128 = add nuw nsw i32 %127, %75
  %129 = zext i32 %128 to i64
  %130 = mul i64 %50, %129
  %131 = getelementptr inbounds i8, i8 addrspace(1)* %122, i64 %130
  %132 = mul i64 %54, %129
  %133 = getelementptr inbounds i8, i8 addrspace(1)* %2, i64 %132
  %134 = bitcast i8 addrspace(1)* %133 to bfloat addrspace(1)*
  %135 = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %132
  %136 = bitcast i8 addrspace(1)* %135 to bfloat addrspace(1)*
  %137 = getelementptr inbounds bfloat, bfloat addrspace(1)* %134, i64 %121
  %138 = load bfloat, bfloat addrspace(1)* %137, align 2, !tbaa !55
  %139 = fpext bfloat %138 to float
  %140 = getelementptr inbounds bfloat, bfloat addrspace(1)* %136, i64 %121
  %141 = load bfloat, bfloat addrspace(1)* %140, align 2, !tbaa !55
  %142 = fpext bfloat %141 to float
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %100) #13
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %101) #13
  call void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %131, float* noundef nonnull %98, float* noundef nonnull %99, float noundef %139, float noundef %142, float noundef %115, float noundef %117, float* noundef nonnull align 4 dereferenceable(4) %15, float* noundef nonnull align 4 dereferenceable(4) %16) #12
  %143 = load float, float* %15, align 4, !tbaa !57
  %144 = zext i32 %127 to i64
  %145 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %144
  %146 = load float, float* %145, align 4, !tbaa !57
  %147 = fadd float %143, %146
  store float %147, float* %145, align 4, !tbaa !57
  %148 = load float, float* %16, align 4, !tbaa !57
  %149 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %144
  %150 = load float, float* %149, align 4, !tbaa !57
  %151 = fadd float %148, %150
  store float %151, float* %149, align 4, !tbaa !57
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %101) #13
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %100) #13
  %152 = add nuw nsw i32 %127, 1
  %153 = icmp eq i32 %152, 4
  br i1 %153, label %123, label %126, !llvm.loop !68

154:                                              ; preds = %195
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %96) #13
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %95) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %94) #13
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %93) #13
  br label %198

155:                                              ; preds = %195, %102
  %156 = phi i16 [ 0, %102 ], [ %196, %195 ]
  %157 = zext i16 %156 to i64
  %158 = getelementptr inbounds [4 x float], [4 x float]* %13, i64 0, i64 %157
  %159 = load float, float* %158, align 4, !tbaa !57
  %160 = call fast float @air.simd_sum.f32(float %159) #14
  br i1 %103, label %161, label %176

161:                                              ; preds = %155
  %162 = fptrunc float %160 to bfloat
  %163 = bitcast float %160 to i32
  %164 = and i32 %163, 2139095040
  %165 = icmp eq i32 %164, 2139095040
  br i1 %165, label %171, label %166

166:                                              ; preds = %161
  %167 = fpext bfloat %162 to float
  %168 = bitcast float %167 to i32
  %169 = and i32 %168, 2139095040
  %170 = icmp eq i32 %169, 2139095040
  br i1 %170, label %171, label %173

171:                                              ; preds = %166, %161
  %172 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %104, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %173

173:                                              ; preds = %171, %166
  %174 = add nuw nsw i64 %107, %157
  %175 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %174
  store bfloat %162, bfloat addrspace(1)* %175, align 2, !tbaa !55
  br label %176

176:                                              ; preds = %173, %155
  %177 = getelementptr inbounds [4 x float], [4 x float]* %14, i64 0, i64 %157
  %178 = load float, float* %177, align 4, !tbaa !57
  %179 = call fast float @air.simd_sum.f32(float %178) #14
  br i1 %103, label %180, label %195

180:                                              ; preds = %176
  %181 = fptrunc float %179 to bfloat
  %182 = bitcast float %179 to i32
  %183 = and i32 %182, 2139095040
  %184 = icmp eq i32 %183, 2139095040
  br i1 %184, label %190, label %185

185:                                              ; preds = %180
  %186 = fpext bfloat %181 to float
  %187 = bitcast float %186 to i32
  %188 = and i32 %187, 2139095040
  %189 = icmp eq i32 %188, 2139095040
  br i1 %189, label %190, label %192

190:                                              ; preds = %185, %180
  %191 = call i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture %104, i32 4, i32 0, i32 2, i32 0, i1 false) #10
  br label %192

192:                                              ; preds = %190, %185
  %193 = add nuw nsw i64 %109, %157
  %194 = getelementptr inbounds bfloat, bfloat addrspace(1)* %4, i64 %193
  store bfloat %181, bfloat addrspace(1)* %194, align 2, !tbaa !55
  br label %195

195:                                              ; preds = %192, %176
  %196 = add nuw nsw i16 %156, 1
  %197 = icmp eq i16 %196, 4
  br i1 %197, label %154, label %155, !llvm.loop !69

198:                                              ; preds = %154, %71, %68, %66
  ret void
}

; Function Attrs: mustprogress nofree nosync nounwind readnone willreturn
declare i1 @air.any.v3i1(<3 x i1>) local_unnamed_addr #2

; Function Attrs: mustprogress nounwind willreturn
declare i32 @air.atomic.global.or.u.i32(i32 addrspace(1)* nocapture, i32, i32, i32, i32, i1) local_unnamed_addr #3

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #4

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt4ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !37
  switch i32 %3, label %25 [
    i32 2560, label %4
    i32 6144, label %28
  ]

4:                                                ; preds = %1
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !43
  switch i32 %6, label %23 [
    i32 10240, label %7
    i32 12288, label %15
  ]

7:                                                ; preds = %4
  %8 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %9 = load i64, i64 addrspace(2)* %8, align 8, !tbaa !50
  %10 = icmp ult i64 %9, 1801615789990190
  %11 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %12 = load i64, i64 addrspace(2)* %11, align 8
  %13 = icmp ult i64 %12, 1801615789990190
  %14 = select i1 %10, i1 %13, i1 false
  br label %62

15:                                               ; preds = %4
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %17 = load i64, i64 addrspace(2)* %16, align 8, !tbaa !50
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
  %27 = load i32, i32 addrspace(2)* %26, align 4, !tbaa !43
  br label %40

28:                                               ; preds = %23, %1
  %29 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %30 = load i32, i32 addrspace(2)* %29, align 4, !tbaa !43
  %31 = icmp eq i32 %30, 2560
  br i1 %31, label %32, label %40

32:                                               ; preds = %28
  %33 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %34 = load i64, i64 addrspace(2)* %33, align 8, !tbaa !50
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
  %45 = load i64, i64 addrspace(2)* %44, align 8, !tbaa !50
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
  %59 = load i64, i64 addrspace(2)* %58, align 8, !tbaa !51
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
define linkonce_odr float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi4EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #5 {
  br label %4

3:                                                ; preds = %4
  ret float %29

4:                                                ; preds = %4, %2
  %5 = phi i32 [ 0, %2 ], [ %37, %4 ]
  %6 = phi float [ 0.000000e+00, %2 ], [ %29, %4 ]
  %7 = zext i32 %5 to i64
  %8 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %7
  %9 = load bfloat, bfloat addrspace(1)* %8, align 2, !tbaa !55
  %10 = fpext bfloat %9 to float
  %11 = or i32 %5, 1
  %12 = zext i32 %11 to i64
  %13 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %12
  %14 = load bfloat, bfloat addrspace(1)* %13, align 2, !tbaa !55
  %15 = fpext bfloat %14 to float
  %16 = fadd float %10, %15
  %17 = or i32 %5, 2
  %18 = zext i32 %17 to i64
  %19 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %18
  %20 = load bfloat, bfloat addrspace(1)* %19, align 2, !tbaa !55
  %21 = fpext bfloat %20 to float
  %22 = fadd float %16, %21
  %23 = or i32 %5, 3
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %24
  %26 = load bfloat, bfloat addrspace(1)* %25, align 2, !tbaa !55
  %27 = fpext bfloat %26 to float
  %28 = fadd float %22, %27
  %29 = fadd float %6, %28
  %30 = getelementptr inbounds float, float* %1, i64 %7
  store float %10, float* %30, align 4, !tbaa !57
  %31 = fmul float %15, 6.250000e-02
  %32 = getelementptr inbounds float, float* %1, i64 %12
  store float %31, float* %32, align 4, !tbaa !57
  %33 = fmul float %21, 3.906250e-03
  %34 = getelementptr inbounds float, float* %1, i64 %18
  store float %33, float* %34, align 4, !tbaa !57
  %35 = fmul float %27, 0x3F30000000000000
  %36 = getelementptr inbounds float, float* %1, i64 %24
  store float %35, float* %36, align 4, !tbaa !57
  %37 = add nuw nsw i32 %5, 4
  %38 = icmp ult i32 %5, 12
  br i1 %38, label %4, label %3, !llvm.loop !70
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt4ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #5 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %59, float %11)
  store float %12, float* %7, align 4, !tbaa !57
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %72, float %13)
  store float %14, float* %8, align 4, !tbaa !57
  ret void

15:                                               ; preds = %15, %9
  %16 = phi float [ 0.000000e+00, %9 ], [ %59, %15 ]
  %17 = phi float [ 0.000000e+00, %9 ], [ %72, %15 ]
  %18 = phi i32 [ 0, %9 ], [ %73, %15 ]
  %19 = shl nuw nsw i32 %18, 1
  %20 = zext i32 %19 to i64
  %21 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %20
  %22 = load i8, i8 addrspace(1)* %21, align 1, !tbaa !71
  %23 = or i32 %19, 1
  %24 = zext i32 %23 to i64
  %25 = getelementptr inbounds i8, i8 addrspace(1)* %0, i64 %24
  %26 = load i8, i8 addrspace(1)* %25, align 1, !tbaa !71
  %27 = zext i8 %26 to i32
  %28 = shl nuw nsw i32 %27, 8
  %29 = and i8 %22, 15
  %30 = and i8 %22, -16
  %31 = and i32 %28, 3840
  %32 = and i32 %28, 61440
  %33 = shl nuw nsw i32 %18, 2
  %34 = zext i32 %33 to i64
  %35 = getelementptr inbounds float, float* %1, i64 %34
  %36 = load float, float* %35, align 4, !tbaa !57
  %37 = zext i8 %29 to i32
  %38 = tail call float @air.convert.f.f32.s.i32(i32 %37) #9
  %39 = or i32 %33, 1
  %40 = zext i32 %39 to i64
  %41 = getelementptr inbounds float, float* %1, i64 %40
  %42 = load float, float* %41, align 4, !tbaa !57
  %43 = zext i8 %30 to i32
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %43) #9
  %45 = fmul float %42, %44
  %46 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %45)
  %47 = or i32 %33, 2
  %48 = zext i32 %47 to i64
  %49 = getelementptr inbounds float, float* %1, i64 %48
  %50 = load float, float* %49, align 4, !tbaa !57
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %31) #9
  %52 = tail call float @llvm.fmuladd.f32(float %50, float %51, float %46)
  %53 = or i32 %33, 3
  %54 = zext i32 %53 to i64
  %55 = getelementptr inbounds float, float* %1, i64 %54
  %56 = load float, float* %55, align 4, !tbaa !57
  %57 = tail call float @air.convert.f.f32.s.i32(i32 %32) #9
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %52)
  %59 = fadd float %16, %58
  %60 = getelementptr inbounds float, float* %2, i64 %34
  %61 = load float, float* %60, align 4, !tbaa !57
  %62 = getelementptr inbounds float, float* %2, i64 %40
  %63 = load float, float* %62, align 4, !tbaa !57
  %64 = fmul float %44, %63
  %65 = tail call float @llvm.fmuladd.f32(float %61, float %38, float %64)
  %66 = getelementptr inbounds float, float* %2, i64 %48
  %67 = load float, float* %66, align 4, !tbaa !57
  %68 = tail call float @llvm.fmuladd.f32(float %67, float %51, float %65)
  %69 = getelementptr inbounds float, float* %2, i64 %54
  %70 = load float, float* %69, align 4, !tbaa !57
  %71 = tail call float @llvm.fmuladd.f32(float %70, float %57, float %68)
  %72 = fadd float %17, %71
  %73 = add nuw nsw i32 %18, 1
  %74 = icmp eq i32 %73, 4
  br i1 %74, label %10, label %15, !llvm.loop !72
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
define linkonce_odr zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt5ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !37
  %4 = icmp eq i32 %3, 2560
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !43
  %7 = icmp eq i32 %6, 10240
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %9, label %17

9:                                                ; preds = %1
  %10 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %11 = load i64, i64 addrspace(2)* %10, align 8, !tbaa !50
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
  %21 = load i64, i64 addrspace(2)* %20, align 8, !tbaa !50
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
  %36 = load i64, i64 addrspace(2)* %35, align 8, !tbaa !51
  %37 = udiv i64 %34, %27
  %38 = icmp ule i64 %36, %37
  br label %39

39:                                               ; preds = %30, %19, %17, %9
  %40 = phi i1 [ %16, %9 ], [ false, %17 ], [ false, %19 ], [ %38, %30 ]
  ret i1 %40
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi16ELi5EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #5 {
  br label %4

3:                                                ; preds = %4
  ret float %54

4:                                                ; preds = %4, %2
  %5 = phi i1 [ true, %2 ], [ false, %4 ]
  %6 = phi i32 [ 0, %2 ], [ 8, %4 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %54, %4 ]
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %8
  %10 = load bfloat, bfloat addrspace(1)* %9, align 2, !tbaa !55
  %11 = fpext bfloat %10 to float
  %12 = or i32 %6, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %13
  %15 = load bfloat, bfloat addrspace(1)* %14, align 2, !tbaa !55
  %16 = fpext bfloat %15 to float
  %17 = fadd float %11, %16
  %18 = or i32 %6, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %19
  %21 = load bfloat, bfloat addrspace(1)* %20, align 2, !tbaa !55
  %22 = fpext bfloat %21 to float
  %23 = fadd float %17, %22
  %24 = or i32 %6, 3
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %25
  %27 = load bfloat, bfloat addrspace(1)* %26, align 2, !tbaa !55
  %28 = fpext bfloat %27 to float
  %29 = fadd float %23, %28
  %30 = or i32 %6, 4
  %31 = zext i32 %30 to i64
  %32 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %31
  %33 = load bfloat, bfloat addrspace(1)* %32, align 2, !tbaa !55
  %34 = fpext bfloat %33 to float
  %35 = fadd float %29, %34
  %36 = or i32 %6, 5
  %37 = zext i32 %36 to i64
  %38 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %37
  %39 = load bfloat, bfloat addrspace(1)* %38, align 2, !tbaa !55
  %40 = fpext bfloat %39 to float
  %41 = fadd float %35, %40
  %42 = or i32 %6, 6
  %43 = zext i32 %42 to i64
  %44 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %43
  %45 = load bfloat, bfloat addrspace(1)* %44, align 2, !tbaa !55
  %46 = fpext bfloat %45 to float
  %47 = fadd float %41, %46
  %48 = or i32 %6, 7
  %49 = zext i32 %48 to i64
  %50 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %49
  %51 = load bfloat, bfloat addrspace(1)* %50, align 2, !tbaa !55
  %52 = fpext bfloat %51 to float
  %53 = fadd float %47, %52
  %54 = fadd float %7, %53
  %55 = getelementptr inbounds float, float* %1, i64 %8
  store float %11, float* %55, align 4, !tbaa !57
  %56 = fmul float %16, 3.125000e-02
  %57 = getelementptr inbounds float, float* %1, i64 %13
  store float %56, float* %57, align 4, !tbaa !57
  %58 = fmul float %22, 2.500000e-01
  %59 = getelementptr inbounds float, float* %1, i64 %19
  store float %58, float* %59, align 4, !tbaa !57
  %60 = fmul float %28, 7.812500e-03
  %61 = getelementptr inbounds float, float* %1, i64 %25
  store float %60, float* %61, align 4, !tbaa !57
  %62 = fmul float %34, 6.250000e-02
  %63 = getelementptr inbounds float, float* %1, i64 %31
  store float %62, float* %63, align 4, !tbaa !57
  %64 = fmul float %40, 5.000000e-01
  %65 = getelementptr inbounds float, float* %1, i64 %37
  store float %64, float* %65, align 4, !tbaa !57
  %66 = fmul float %46, 1.562500e-02
  %67 = getelementptr inbounds float, float* %1, i64 %43
  store float %66, float* %67, align 4, !tbaa !57
  %68 = fmul float %52, 1.250000e-01
  %69 = getelementptr inbounds float, float* %1, i64 %49
  store float %68, float* %69, align 4, !tbaa !57
  br i1 %5, label %4, label %3, !llvm.loop !73
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt5ELt16EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #5 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %98, float %11)
  store float %12, float* %7, align 4, !tbaa !57
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %129, float %13)
  store float %14, float* %8, align 4, !tbaa !57
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
  %30 = load i8, i8 addrspace(1)* %29, align 1, !tbaa !71
  %31 = zext i8 %30 to i32
  %32 = and i32 %31, 31
  %33 = and i32 %31, 224
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 1
  %35 = load i8, i8 addrspace(1)* %34, align 1, !tbaa !71
  %36 = zext i8 %35 to i32
  %37 = and i32 %36, 3
  %38 = and i32 %36, 124
  %39 = and i32 %36, 128
  %40 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 2
  %41 = load i8, i8 addrspace(1)* %40, align 1, !tbaa !71
  %42 = zext i8 %41 to i32
  %43 = and i32 %42, 15
  %44 = and i32 %42, 240
  %45 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 3
  %46 = load i8, i8 addrspace(1)* %45, align 1, !tbaa !71
  %47 = zext i8 %46 to i32
  %48 = and i32 %47, 1
  %49 = and i32 %47, 62
  %50 = and i32 %47, 192
  %51 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 4
  %52 = load i8, i8 addrspace(1)* %51, align 1, !tbaa !71
  %53 = zext i8 %52 to i32
  %54 = and i32 %53, 7
  %55 = and i32 %53, 248
  %56 = tail call float @air.convert.f.f32.s.i32(i32 %32) #9
  %57 = load float, float* %25, align 4, !tbaa !57
  %58 = tail call float @llvm.fmuladd.f32(float %56, float %57, float %19)
  %59 = tail call float @air.convert.f.f32.s.i32(i32 %33) #9
  %60 = getelementptr inbounds float, float* %25, i64 1
  %61 = load float, float* %60, align 4, !tbaa !57
  %62 = tail call float @llvm.fmuladd.f32(float %59, float %61, float %58)
  %63 = tail call float @air.convert.f.f32.s.i32(i32 %37) #9
  %64 = fmul float %61, 2.560000e+02
  %65 = tail call float @llvm.fmuladd.f32(float %63, float %64, float %62)
  %66 = tail call float @air.convert.f.f32.s.i32(i32 %38) #9
  %67 = getelementptr inbounds float, float* %25, i64 2
  %68 = load float, float* %67, align 4, !tbaa !57
  %69 = tail call float @llvm.fmuladd.f32(float %66, float %68, float %65)
  %70 = tail call float @air.convert.f.f32.s.i32(i32 %39) #9
  %71 = getelementptr inbounds float, float* %25, i64 3
  %72 = load float, float* %71, align 4, !tbaa !57
  %73 = tail call float @llvm.fmuladd.f32(float %70, float %72, float %69)
  %74 = tail call float @air.convert.f.f32.s.i32(i32 %43) #9
  %75 = fmul float %72, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %74, float %75, float %73)
  %77 = tail call float @air.convert.f.f32.s.i32(i32 %44) #9
  %78 = getelementptr inbounds float, float* %25, i64 4
  %79 = load float, float* %78, align 4, !tbaa !57
  %80 = tail call float @llvm.fmuladd.f32(float %77, float %79, float %76)
  %81 = tail call float @air.convert.f.f32.s.i32(i32 %48) #9
  %82 = fmul float %79, 2.560000e+02
  %83 = tail call float @llvm.fmuladd.f32(float %81, float %82, float %80)
  %84 = tail call float @air.convert.f.f32.s.i32(i32 %49) #9
  %85 = getelementptr inbounds float, float* %25, i64 5
  %86 = load float, float* %85, align 4, !tbaa !57
  %87 = tail call float @llvm.fmuladd.f32(float %84, float %86, float %83)
  %88 = tail call float @air.convert.f.f32.s.i32(i32 %50) #9
  %89 = getelementptr inbounds float, float* %25, i64 6
  %90 = load float, float* %89, align 4, !tbaa !57
  %91 = tail call float @llvm.fmuladd.f32(float %88, float %90, float %87)
  %92 = tail call float @air.convert.f.f32.s.i32(i32 %54) #9
  %93 = fmul float %90, 2.560000e+02
  %94 = tail call float @llvm.fmuladd.f32(float %92, float %93, float %91)
  %95 = tail call float @air.convert.f.f32.s.i32(i32 %55) #9
  %96 = getelementptr inbounds float, float* %25, i64 7
  %97 = load float, float* %96, align 4, !tbaa !57
  %98 = tail call float @llvm.fmuladd.f32(float %95, float %97, float %94)
  %99 = load float, float* %26, align 4, !tbaa !57
  %100 = tail call float @llvm.fmuladd.f32(float %56, float %99, float %20)
  %101 = getelementptr inbounds float, float* %26, i64 1
  %102 = load float, float* %101, align 4, !tbaa !57
  %103 = tail call float @llvm.fmuladd.f32(float %59, float %102, float %100)
  %104 = fmul float %102, 2.560000e+02
  %105 = tail call float @llvm.fmuladd.f32(float %63, float %104, float %103)
  %106 = getelementptr inbounds float, float* %26, i64 2
  %107 = load float, float* %106, align 4, !tbaa !57
  %108 = tail call float @llvm.fmuladd.f32(float %66, float %107, float %105)
  %109 = getelementptr inbounds float, float* %26, i64 3
  %110 = load float, float* %109, align 4, !tbaa !57
  %111 = tail call float @llvm.fmuladd.f32(float %70, float %110, float %108)
  %112 = fmul float %110, 2.560000e+02
  %113 = tail call float @llvm.fmuladd.f32(float %74, float %112, float %111)
  %114 = getelementptr inbounds float, float* %26, i64 4
  %115 = load float, float* %114, align 4, !tbaa !57
  %116 = tail call float @llvm.fmuladd.f32(float %77, float %115, float %113)
  %117 = fmul float %115, 2.560000e+02
  %118 = tail call float @llvm.fmuladd.f32(float %81, float %117, float %116)
  %119 = getelementptr inbounds float, float* %26, i64 5
  %120 = load float, float* %119, align 4, !tbaa !57
  %121 = tail call float @llvm.fmuladd.f32(float %84, float %120, float %118)
  %122 = getelementptr inbounds float, float* %26, i64 6
  %123 = load float, float* %122, align 4, !tbaa !57
  %124 = tail call float @llvm.fmuladd.f32(float %88, float %123, float %121)
  %125 = fmul float %123, 2.560000e+02
  %126 = tail call float @llvm.fmuladd.f32(float %92, float %125, float %124)
  %127 = getelementptr inbounds float, float* %26, i64 7
  %128 = load float, float* %127, align 4, !tbaa !57
  %129 = tail call float @llvm.fmuladd.f32(float %95, float %128, float %126)
  br i1 %21, label %15, label %10, !llvm.loop !74
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt5ELt128EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !37
  switch i32 %3, label %4 [
    i32 6144, label %11
    i32 2560, label %7
  ]

4:                                                ; preds = %1
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !43
  br label %31

7:                                                ; preds = %1
  %8 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %9 = load i32, i32 addrspace(2)* %8, align 4, !tbaa !43
  %10 = icmp eq i32 %9, 6144
  br i1 %10, label %23, label %31

11:                                               ; preds = %1
  %12 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %13 = load i32, i32 addrspace(2)* %12, align 4, !tbaa !43
  %14 = icmp eq i32 %13, 2560
  br i1 %14, label %15, label %31

15:                                               ; preds = %11
  %16 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %17 = load i64, i64 addrspace(2)* %16, align 8, !tbaa !50
  %18 = icmp ult i64 %17, 7208575253501192
  %19 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 10
  %20 = load i64, i64 addrspace(2)* %19, align 8
  %21 = icmp ult i64 %20, 7208575253501193
  %22 = select i1 %18, i1 %21, i1 false
  br label %54

23:                                               ; preds = %7
  %24 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %25 = load i64, i64 addrspace(2)* %24, align 8, !tbaa !50
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
  %36 = load i64, i64 addrspace(2)* %35, align 8, !tbaa !50
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
  %51 = load i64, i64 addrspace(2)* %50, align 8, !tbaa !51
  %52 = udiv i64 %49, %42
  %53 = icmp ule i64 %51, %52
  br label %54

54:                                               ; preds = %45, %34, %31, %23, %15
  %55 = phi i1 [ %22, %15 ], [ %30, %23 ], [ false, %31 ], [ false, %34 ], [ %53, %45 ]
  ret i1 %55
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr zeroext i1 @_ZN24r5_raw_odd_rowpair_sep2217row_stride_boundsILt6ELt64EEEbRU11MTLconstantK17FlashAffineParams(%struct.FlashAffineParams addrspace(2)* noundef align 8 dereferenceable(64) %0) local_unnamed_addr #5 {
  %2 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 2
  %3 = load i32, i32 addrspace(2)* %2, align 8, !tbaa !37
  %4 = icmp eq i32 %3, 2560
  %5 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 3
  %6 = load i32, i32 addrspace(2)* %5, align 4, !tbaa !43
  %7 = icmp eq i32 %6, 6144
  %8 = select i1 %4, i1 %7, i1 false
  br i1 %8, label %9, label %17

9:                                                ; preds = %1
  %10 = getelementptr inbounds %struct.FlashAffineParams, %struct.FlashAffineParams addrspace(2)* %0, i64 0, i32 8
  %11 = load i64, i64 addrspace(2)* %10, align 8, !tbaa !50
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
  %21 = load i64, i64 addrspace(2)* %20, align 8, !tbaa !50
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
  %36 = load i64, i64 addrspace(2)* %35, align 8, !tbaa !51
  %37 = udiv i64 %34, %27
  %38 = icmp ule i64 %36, %37
  br label %39

39:                                               ; preds = %30, %19, %17, %9
  %40 = phi i1 [ %16, %9 ], [ false, %17 ], [ false, %19 ], [ %38, %30 ]
  ret i1 %40
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr float @_ZN18r5_raw_odd_literal30mlx_qmv_f32xsum_v1_load_vectorIDF16bfLi8ELi6EEET0_PU9MTLdeviceKT_PU9MTLthreadS1_(bfloat addrspace(1)* noundef %0, float* noundef %1) local_unnamed_addr #5 {
  br label %4

3:                                                ; preds = %4
  ret float %30

4:                                                ; preds = %4, %2
  %5 = phi i1 [ true, %2 ], [ false, %4 ]
  %6 = phi i32 [ 0, %2 ], [ 4, %4 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %30, %4 ]
  %8 = zext i32 %6 to i64
  %9 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %8
  %10 = load bfloat, bfloat addrspace(1)* %9, align 2, !tbaa !55
  %11 = fpext bfloat %10 to float
  %12 = or i32 %6, 1
  %13 = zext i32 %12 to i64
  %14 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %13
  %15 = load bfloat, bfloat addrspace(1)* %14, align 2, !tbaa !55
  %16 = fpext bfloat %15 to float
  %17 = fadd float %11, %16
  %18 = or i32 %6, 2
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %19
  %21 = load bfloat, bfloat addrspace(1)* %20, align 2, !tbaa !55
  %22 = fpext bfloat %21 to float
  %23 = fadd float %17, %22
  %24 = or i32 %6, 3
  %25 = zext i32 %24 to i64
  %26 = getelementptr inbounds bfloat, bfloat addrspace(1)* %0, i64 %25
  %27 = load bfloat, bfloat addrspace(1)* %26, align 2, !tbaa !55
  %28 = fpext bfloat %27 to float
  %29 = fadd float %23, %28
  %30 = fadd float %7, %29
  %31 = getelementptr inbounds float, float* %1, i64 %8
  store float %11, float* %31, align 4, !tbaa !57
  %32 = fmul float %16, 1.562500e-02
  %33 = getelementptr inbounds float, float* %1, i64 %13
  store float %32, float* %33, align 4, !tbaa !57
  %34 = fmul float %22, 6.250000e-02
  %35 = getelementptr inbounds float, float* %1, i64 %19
  store float %34, float* %35, align 4, !tbaa !57
  %36 = fmul float %28, 2.500000e-01
  %37 = getelementptr inbounds float, float* %1, i64 %25
  store float %36, float* %37, align 4, !tbaa !57
  br i1 %5, label %4, label %3, !llvm.loop !75
}

; Function Attrs: inlinehint mustprogress nounwind
define linkonce_odr void @_ZN24r5_raw_odd_rowpair_sep229qdot_pairILt6ELt8EEEvPU9MTLdeviceKhPU9MTLthreadKfS4_ffffRU9MTLthreadfS6_(i8 addrspace(1)* noundef %0, float* noundef %1, float* noundef %2, float noundef %3, float noundef %4, float noundef %5, float noundef %6, float* noundef nonnull align 4 dereferenceable(4) %7, float* noundef nonnull align 4 dereferenceable(4) %8) local_unnamed_addr #5 {
  br label %15

10:                                               ; preds = %15
  %11 = fmul float %4, %5
  %12 = tail call float @llvm.fmuladd.f32(float %3, float %64, float %11)
  store float %12, float* %7, align 4, !tbaa !57
  %13 = fmul float %4, %6
  %14 = tail call float @llvm.fmuladd.f32(float %3, float %79, float %13)
  store float %14, float* %8, align 4, !tbaa !57
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
  %30 = load i8, i8 addrspace(1)* %29, align 1, !tbaa !71
  %31 = zext i8 %30 to i32
  %32 = and i32 %31, 63
  %33 = and i32 %31, 192
  %34 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 1
  %35 = load i8, i8 addrspace(1)* %34, align 1, !tbaa !71
  %36 = zext i8 %35 to i32
  %37 = and i32 %36, 15
  %38 = and i32 %36, 240
  %39 = getelementptr inbounds i8, i8 addrspace(1)* %29, i64 2
  %40 = load i8, i8 addrspace(1)* %39, align 1, !tbaa !71
  %41 = zext i8 %40 to i32
  %42 = and i32 %41, 3
  %43 = and i32 %41, 252
  %44 = tail call float @air.convert.f.f32.s.i32(i32 %32) #9
  %45 = load float, float* %25, align 4, !tbaa !57
  %46 = tail call float @llvm.fmuladd.f32(float %44, float %45, float %19)
  %47 = tail call float @air.convert.f.f32.s.i32(i32 %33) #9
  %48 = getelementptr inbounds float, float* %25, i64 1
  %49 = load float, float* %48, align 4, !tbaa !57
  %50 = tail call float @llvm.fmuladd.f32(float %47, float %49, float %46)
  %51 = tail call float @air.convert.f.f32.s.i32(i32 %37) #9
  %52 = fmul float %49, 2.560000e+02
  %53 = tail call float @llvm.fmuladd.f32(float %51, float %52, float %50)
  %54 = tail call float @air.convert.f.f32.s.i32(i32 %38) #9
  %55 = getelementptr inbounds float, float* %25, i64 2
  %56 = load float, float* %55, align 4, !tbaa !57
  %57 = tail call float @llvm.fmuladd.f32(float %54, float %56, float %53)
  %58 = tail call float @air.convert.f.f32.s.i32(i32 %42) #9
  %59 = fmul float %56, 2.560000e+02
  %60 = tail call float @llvm.fmuladd.f32(float %58, float %59, float %57)
  %61 = tail call float @air.convert.f.f32.s.i32(i32 %43) #9
  %62 = getelementptr inbounds float, float* %25, i64 3
  %63 = load float, float* %62, align 4, !tbaa !57
  %64 = tail call float @llvm.fmuladd.f32(float %61, float %63, float %60)
  %65 = load float, float* %26, align 4, !tbaa !57
  %66 = tail call float @llvm.fmuladd.f32(float %44, float %65, float %20)
  %67 = getelementptr inbounds float, float* %26, i64 1
  %68 = load float, float* %67, align 4, !tbaa !57
  %69 = tail call float @llvm.fmuladd.f32(float %47, float %68, float %66)
  %70 = fmul float %68, 2.560000e+02
  %71 = tail call float @llvm.fmuladd.f32(float %51, float %70, float %69)
  %72 = getelementptr inbounds float, float* %26, i64 2
  %73 = load float, float* %72, align 4, !tbaa !57
  %74 = tail call float @llvm.fmuladd.f32(float %54, float %73, float %71)
  %75 = fmul float %73, 2.560000e+02
  %76 = tail call float @llvm.fmuladd.f32(float %58, float %75, float %74)
  %77 = getelementptr inbounds float, float* %26, i64 3
  %78 = load float, float* %77, align 4, !tbaa !57
  %79 = tail call float @llvm.fmuladd.f32(float %61, float %78, float %76)
  br i1 %21, label %15, label %10, !llvm.loop !76
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
!air.kernel = !{!9, !27, !28, !29}
!air.compile_options = !{!30, !31, !32}
!llvm.ident = !{!33}
!air.version = !{!34}
!air.language_version = !{!35}
!air.source_file_name = !{!36}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_timed_q4_g64, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !20, !22, !23, !24, !25, !26}
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
!24 = !{i32 10, !"air.threads_per_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_width"}
!25 = !{i32 11, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
!26 = !{i32 12, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!27 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_timed_q5_g64, !10, !11}
!28 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_timed_q5_g128, !10, !11}
!29 = !{void (bfloat addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i64 addrspace(1)*, bfloat addrspace(1)*, %"struct.metal::_atomic" addrspace(1)*, %struct.FlashAffineParams addrspace(2)*, <3 x i32>, <3 x i32>, i32, i32, i32)* @raw_large_R4_guard_pair_sep22_timed_q6_g64, !10, !11}
!30 = !{!"air.compile.denorms_disable"}
!31 = !{!"air.compile.fast_math_enable"}
!32 = !{!"air.compile.framebuffer_fetch_enable"}
!33 = !{!"Apple metal version 32023.921 (metalfe-32023.921.6)"}
!34 = !{i32 2, i32 9, i32 0}
!35 = !{!"Metal", i32 4, i32 1, i32 0}
!36 = !{!"/Users/mweinbach/Projects/splash/dev/benchmarks/raw_large_R4_guard_pair_sep22/kernel/candidate.metal"}
!37 = !{!38, !39, i64 8}
!38 = !{!"_ZTS17FlashAffineParams", !39, i64 0, !39, i64 4, !39, i64 8, !39, i64 12, !39, i64 16, !39, i64 20, !39, i64 24, !39, i64 28, !42, i64 32, !42, i64 40, !42, i64 48, !42, i64 56}
!39 = !{!"int", !40, i64 0}
!40 = !{!"omnipotent char", !41, i64 0}
!41 = !{!"Simple C++ TBAA"}
!42 = !{!"long", !40, i64 0}
!43 = !{!38, !39, i64 12}
!44 = !{!38, !39, i64 0}
!45 = !{!38, !39, i64 4}
!46 = !{!38, !39, i64 16}
!47 = !{!38, !39, i64 28}
!48 = !{!38, !39, i64 20}
!49 = !{!38, !39, i64 24}
!50 = !{!38, !42, i64 32}
!51 = !{!38, !42, i64 48}
!52 = !{!38, !42, i64 56}
!53 = distinct !{!53, !54}
!54 = !{!"llvm.loop.mustprogress"}
!55 = !{!56, !56, i64 0}
!56 = !{!"bfloat", !40, i64 0}
!57 = !{!58, !58, i64 0}
!58 = !{!"float", !40, i64 0}
!59 = distinct !{!59, !54}
!60 = distinct !{!60, !54}
!61 = distinct !{!61, !54}
!62 = distinct !{!62, !54}
!63 = distinct !{!63, !54}
!64 = distinct !{!64, !54}
!65 = distinct !{!65, !54}
!66 = distinct !{!66, !54}
!67 = distinct !{!67, !54}
!68 = distinct !{!68, !54}
!69 = distinct !{!69, !54}
!70 = distinct !{!70, !54}
!71 = !{!40, !40, i64 0}
!72 = distinct !{!72, !54}
!73 = distinct !{!73, !54}
!74 = distinct !{!74, !54}
!75 = distinct !{!75, !54}
!76 = distinct !{!76, !54}
