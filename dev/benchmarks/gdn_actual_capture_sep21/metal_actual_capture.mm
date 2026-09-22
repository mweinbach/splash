// The source and dispatch below are the sealed v6 oracle, unchanged. No Metal
// device or binary tensor is opened by default/help/--cpu-contract.
#define main sealed_v6_oracle_main
#include "../gdn_nax_chunks_sep21_v6/metal_oracle.mm"
#undef main

namespace {
constexpr const char *V6LibrarySHA = "c58a0344a1cb2e6a29d4aef1a81c4eff2f01191a4380321e48323658a867b709";
constexpr Variant ActualVariant{"private_gdn_wy_v32_t32_sg8",32,32,8};
const std::map<std::string,size_t> CaptureExtents{
    {"mixed",2048*Mixed*2},{"decay",2048*Heads*4},{"beta",2048*Heads*2},
    {"z",2048*Out*2},{"norm",256},{"initial_state",State*4},
    {"expected_state",State*4},{"expected_recurrence",2048*Out*2},
    {"expected_output",2048*Out*2}};

struct ActualCapture {
    NSDictionary *manifest;
    std::string manifestHash;
    std::map<std::string,NSData *> data;
    explicit ActualCapture(const std::string &path,bool readPayload) {
        NSError *error=nil;
        NSData *text=[NSData dataWithContentsOfFile:ns(path) options:0 error:&error];
        require(text!=nil,"manifest read: "+str(error.localizedDescription));
        manifestHash=digest(text.bytes,text.length);
        id parsed=[NSJSONSerialization JSONObjectWithData:text options:0 error:&error];
        require([parsed isKindOfClass:[NSDictionary class]],"manifest JSON dictionary required");
        manifest=parsed;
        require([manifest[@"schema"] isEqual:@"splash-actual-gdn-layer-v1"],"capture schema");
        require([manifest[@"layer"] isKindOfClass:[NSNumber class]] &&
            [manifest[@"layer"] unsignedLongLongValue]==0,"capture layer0 required");
        require([manifest[@"rows"] isKindOfClass:[NSNumber class]] &&
            [manifest[@"rows"] unsignedLongLongValue]==2048,"capture R2048 required");
        require([manifest[@"lanes"] isKindOfClass:[NSNumber class]] &&
            [manifest[@"lanes"] unsignedLongLongValue]==1,"capture singleton required");
        require([manifest[@"norm_epsilon"] isKindOfClass:[NSNumber class]] &&
            float([manifest[@"norm_epsilon"] doubleValue])==1e-6f,"capture norm epsilon");
        NSDictionary *files=manifest[@"files"];
        require([files isKindOfClass:[NSDictionary class]] && files.count==CaptureExtents.size(),
            "capture exact file inventory required");
        NSMutableArray *records=[NSMutableArray array];
        for (const auto &[key,bytes]:CaptureExtents) {
            NSDictionary *record=files[ns(key)];
            require([record isKindOfClass:[NSDictionary class]],"missing capture file "+key);
            require([record[@"bytes"] isKindOfClass:[NSNumber class]] &&
                [record[@"bytes"] unsignedLongLongValue]==bytes,"capture extent "+key);
            NSString *payloadPath=record[@"path"],*sha=record[@"sha256"];
            require([payloadPath isKindOfClass:[NSString class]] &&
                std::filesystem::path(str(payloadPath)).is_absolute(),"absolute capture path "+key);
            require([sha isKindOfClass:[NSString class]] && sha.length==64,"capture SHA length "+key);
            for (char c:str(sha)) require((c>='0' && c<='9') || (c>='a' && c<='f'),"capture SHA syntax "+key);
            if (readPayload) {
                NSData *tensor=[NSData dataWithContentsOfFile:payloadPath options:0 error:&error];
                require(tensor!=nil,"capture payload read "+key+": "+str(error.localizedDescription));
                require(tensor.length==bytes,"capture actual extent "+key);
                require(digest(tensor.bytes,tensor.length)==str(sha),"capture content SHA "+key);
                data.emplace(key,tensor);
            }
            [records addObject:@{@"key":ns(key),@"bytes":@(bytes),@"sha256":sha,
                @"content_verified":@(readPayload)}];
        }
        emit(@{@"kind":@"actual_capture_manifest",@"manifest_sha256":ns(manifestHash),
            @"rows":@2048,@"lanes":@1,@"layer":@0,@"files":records,
            @"payload_read":@(readPayload),@"metal_device_created":@NO});
    }
    NSData *tensor(const std::string &key) const {return data.at(key);}
    std::vector<float> state(const std::string &key) const {
        auto *p=static_cast<const float *>(tensor(key).bytes);
        return {p,p+State};
    }
    void immutable() const {
        NSDictionary *files=manifest[@"files"];
        for (const auto &[key,tensor]:data)
            require(digest(tensor.bytes,tensor.length)==str(files[ns(key)][@"sha256"]),
                "loaded capture changed "+key);
    }
};

Inputs actualInputs(GPU &gpu,const ActualCapture &capture) {
    // Allocate through the frozen v6 Inputs type. The temporary synthetic bytes
    // are replaced in full before every immutable hash and before any dispatch.
    Inputs input(gpu,gdn_chunk_cpu::continuation_fixture(2048),1);
    for (auto [key,buffer]:std::array<std::pair<const char *,Guarded *>,3>{{
        {"mixed",&input.mixed},{"decay",&input.decay},{"beta",&input.beta}}}) {
        const NSData *tensor=capture.tensor(key);
        require(tensor.length==buffer->bytes,"actual Inputs extent");
        std::memcpy(buffer->data(),tensor.bytes,tensor.length);
        buffer->freeze();
    }
    FlashGDNParams p{2048,1,16,48,128,128,4,1e-6f,uint64_t(3*Mixed*2),uint64_t(State*4)};
    std::memcpy(input.params.data(),&p,sizeof(p)); input.params.freeze();
    input.check();
    return input;
}

struct ActualQuality {bool pass; std::vector<float> candidateCarry;};
ActualQuality actualQuality(GPU &gpu,const Inputs &input,const std::vector<float> &seed,
    const char *stage,Results &candidate,Results &native) {
    candidate.reset(seed); native.reset(seed);
    dispatch(gpu,gpu.pipeline(Canonical.name),Canonical,input,native,false,false);
    dispatch(gpu,gpu.pipeline(ActualVariant.name),ActualVariant,input,candidate,false,false,
        gpu.pipeline(preparationName(ActualVariant)));
    input.check(); candidate.check(false,seed); native.check(false,seed);
    candidate.seed.immutable(); native.seed.immutable();
    std::vector<double> stateTruth(State),outputTruth(input.rows*Out);
    for (size_t i=0;i<State;++i) stateTruth[i]=native.state.as<float>()[i];
    for (size_t i=0;i<outputTruth.size();++i) outputTruth[i]=from_bf16(native.output.as<uint16_t>()[i]);
    const auto state=errors(State,stateTruth,1,[&](size_t i) {return double(candidate.state.as<float>()[i]);});
    const auto output=errors(outputTruth.size(),outputTruth,1,[&](size_t i) {return double(from_bf16(candidate.output.as<uint16_t>()[i]));});
    const auto requiredState=requiredErrors(State,stateTruth,[&](size_t i) {
        return candidate.flags.as<uint32_t>()[i/(V*K)]==0;
    },[&](size_t i) {return double(candidate.state.as<float>()[i]);});
    const auto requiredOutput=requiredErrors(outputTruth.size(),outputTruth,[&](size_t i) {
        return candidate.flags.as<uint32_t>()[(i%Out)/V]==0;
    },[&](size_t i) {return double(from_bf16(candidate.output.as<uint16_t>()[i]));});
    const auto match=nativeEquivalence(candidate,native,Heads,false);
    const bool pass=state.pass() && bf16Pass(output) && requiredState.pass() &&
        bf16Pass(requiredOutput) && match.pass() && !*candidate.diagnostics.as<uint32_t>() &&
        !*native.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"actual_capture_quality",@"stage":ns(stage),@"variant":ns(ActualVariant.name),
        @"qualification_pass":@(pass),@"state_f32_vs_native":state.json(),
        @"recurrence_bf16_vs_native":bf16Metrics(output),
        @"unflagged_state_f32_vs_native":requiredState.json(),
        @"unflagged_recurrence_bf16_vs_native":bf16Metrics(requiredOutput),
        @"native_replay_equivalence":match.json(),@"eligibility":flagMetrics(candidate,Heads),
        @"candidate_diagnostics":@(*candidate.diagnostics.as<uint32_t>()),
        @"native_diagnostics":@(*native.diagnostics.as<uint32_t>()),
        @"canaries_pass":@YES,@"immutable_input_and_seed_sha256_pass":@YES,
        @"reference":@"frozen original GPU native recurrence from identical incoming F32 seed",
        @"unflagged_wy_bit_exact_claim":@NO,@"unchanged_f32_and_bf16_quality_gates":@YES});
    return {pass,{candidate.state.as<float>(),candidate.state.as<float>()+State}};
}

void actualBench(GPU &gpu,const Inputs &input,const std::vector<float> &seed,const char *stage,
    Results &candidate,Results &native) {
    candidate.reset(seed); native.reset(seed);
    const auto candidatePipeline=gpu.pipeline(ActualVariant.name);
    const auto nativePipeline=gpu.pipeline(Canonical.name);
    const auto preparation=gpu.pipeline(preparationName(ActualVariant));
    auto run=[&](bool useCandidate) {
        return dispatch(gpu,useCandidate?candidatePipeline:nativePipeline,
            useCandidate?ActualVariant:Canonical,input,useCandidate?candidate:native,false,false,
            useCandidate?preparation:nil,true);
    };
    double warmed=0; unsigned commands=0;
    while (warmed<.150 || commands<8) {
        warmed+=run(false).gpu; warmed+=run(true).gpu; commands+=2;
        require(commands<4096,"bounded actual GPU warmup");
    }
    emit(@{@"kind":@"actual_capture_bench_warmup",@"stage":ns(stage),@"gpu_ms":@(warmed*1000),
        @"commands":@(commands),@"minimum_gpu_ms":@150,@"cpu_tensor_access_during_warmup":@NO});
    std::vector<double> nativeTimes,candidateTimes,speedups,wallSpeedups;
    for (unsigned pair=0;pair<10;++pair) {
        // Exactly five ABBA and five BAAB pairs; deterministic balanced order.
        const bool reverse=(pair%2)!=0;
        const std::array<bool,4> order{{reverse,!reverse,!reverse,reverse}};
        std::array<Timing,2> a{},b{}; unsigned ai=0,bi=0;
        for (unsigned slot=0;slot<4;++slot) {
            const Timing t=run(order[slot]);
            if (order[slot]) b[bi++]=t; else a[ai++]=t;
            emit(@{@"kind":@"actual_capture_bench_sample",@"stage":ns(stage),@"pair":@(pair),
                @"slot":@(slot),@"order":reverse?@"BAAB":@"ABBA",
                @"run":order[slot]?@"candidate":@"canonical",@"gpu_ms":@(t.gpu*1000),@"wall_ms":@(t.wall*1000)});
        }
        const double at=(a[0].gpu+a[1].gpu)*.5,bt=(b[0].gpu+b[1].gpu)*.5;
        nativeTimes.push_back(at);candidateTimes.push_back(bt);speedups.push_back(at/bt);
        wallSpeedups.push_back((a[0].wall+a[1].wall)/(b[0].wall+b[1].wall));
    }
    // All CPU tensor inspection starts only after every timed GPU command.
    input.check();candidate.check(false,seed);native.check(false,seed);
    candidate.seed.immutable();native.seed.immutable();
    require(!*candidate.diagnostics.as<uint32_t>() && !*native.diagnostics.as<uint32_t>(),
        "actual timed diagnostics");
    const auto match=nativeEquivalence(candidate,native,Heads,false);
    require(match.pass(),"actual timed flagged native replay not byte exact");
    emit(@{@"kind":@"actual_capture_bench_summary",@"stage":ns(stage),@"rows":@2048,@"lanes":@1,
        @"variant":ns(ActualVariant.name),@"matched_pairs":@10,@"abba_pairs":@5,@"baab_pairs":@5,
        @"canonical_gpu_ms_median":@(median(nativeTimes)*1000),
        @"candidate_gpu_ms_median":@(median(candidateTimes)*1000),
        @"paired_gpu_speedup_median":@(median(speedups)),@"paired_wall_speedup_median":@(median(wallSpeedups)),
        @"eligibility":flagMetrics(candidate,Heads),@"native_replay_equivalence":match.json(),
        @"coefficients_bytes":@(candidate.coefficients.bytes),@"snapshot_bytes":@(candidate.snapshot.bytes),
        @"flags_bytes":@(candidate.flags.bytes),@"snapshot_prepare_wy_restore_native_replay_included":@YES,
        @"gpu_state_reset_copy_included":@YES,@"cpu_tensor_access_during_timing":@NO,
        @"canaries_pass":@YES,@"immutable_input_and_seed_sha256_pass":@YES,
        @"scope":@"captured actual layer0 GDN component; not whole model prefill throughput"});
}

bool actualRun(const ActualCapture &capture,const std::string &library) {
    NSError *error=nil;
    NSData *libraryData=[NSData dataWithContentsOfFile:ns(library) options:0 error:&error];
    require(libraryData!=nil && digest(libraryData.bytes,libraryData.length)==V6LibrarySHA,
        "unchanged sealed v6 library SHA required");
    GPU gpu(library);
    Inputs input=actualInputs(gpu,capture);
    const auto initial=capture.state("initial_state");
    Results candidate(gpu,2048,1,false,32),native(gpu,2048,1,false);
    // Provenance gate runs original native first. Stop before WY if the recorded
    // actual capture cannot be reproduced byte-for-byte by the frozen native.
    native.reset(initial);
    dispatch(gpu,gpu.pipeline(Canonical.name),Canonical,input,native,false,false);
    input.check();native.check(false,initial);native.seed.immutable();
    const size_t stateMismatch=differingBytes(native.state.data(),capture.tensor("expected_state").bytes,State*4);
    const size_t outputMismatch=differingBytes(native.output.data(),capture.tensor("expected_recurrence").bytes,2048*Out*2);
    const bool baselineExact=!stateMismatch && !outputMismatch && !*native.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"actual_capture_native_recorded_provenance",@"byte_exact":@(baselineExact),
        @"state_mismatch_bytes":@(stateMismatch),@"recurrence_mismatch_bytes":@(outputMismatch),
        @"native_diagnostics":@(*native.diagnostics.as<uint32_t>()),@"canaries_pass":@YES,
        @"immutable_input_and_seed_sha256_pass":@YES,@"candidate_executed":@NO});
    require(baselineExact,"recorded actual cold native provenance gate failed");
    const auto cold=actualQuality(gpu,input,initial,"cold",candidate,native);
    const std::vector<uint16_t> coldOutput(candidate.output.as<uint16_t>(),
        candidate.output.as<uint16_t>()+2048*Out);
    const std::vector<uint32_t> coldFlags(candidate.flags.as<uint32_t>(),
        candidate.flags.as<uint32_t>()+Heads);
    const auto nativeCarry=capture.state("expected_state");
    const auto carried=actualQuality(gpu,input,nativeCarry,"carried_same_native_seed",candidate,native);
    // Propagation screen: feed the candidate's own cold carry into both kernels,
    // preserving the identical incoming-state condition for exact native replay.
    const auto propagated=actualQuality(gpu,input,cold.candidateCarry,
        "carried_same_candidate_seed",candidate,native);
    // Diagnostics are sticky OR flags, independently of resettable per-head
    // eligibility. This repeat also checks deterministic flags/state/output.
    candidate.reset(initial);
    *candidate.diagnostics.as<uint32_t>()=0x80000000u;
    dispatch(gpu,gpu.pipeline(ActualVariant.name),ActualVariant,input,candidate,false,false,
        gpu.pipeline(preparationName(ActualVariant)));
    input.check();candidate.check(false,initial);candidate.seed.immutable();
    require(*candidate.diagnostics.as<uint32_t>()==0x80000000u,"sticky diagnostic sentinel lost or extra diagnostic");
    require(!differingBytes(candidate.state.data(),cold.candidateCarry.data(),State*4) &&
        !differingBytes(candidate.output.data(),coldOutput.data(),2048*Out*2) &&
        !differingBytes(candidate.flags.data(),coldFlags.data(),Heads*4),
        "sticky diagnostic repeat changed cold state/output/head flags");
    emit(@{@"kind":@"actual_capture_sticky_diagnostics",@"sticky_sentinel":@(0x80000000u),
        @"preserved":@YES,@"eligibility":flagMetrics(candidate,Heads),@"canaries_pass":@YES,
        @"repeat_state_output_and_headflags_byte_exact":@YES,
        @"note":@"per-head flags reset each snapshot; diagnostic OR sentinel remains sticky"});
    actualBench(gpu,input,initial,"cold",candidate,native);
    actualBench(gpu,input,nativeCarry,"carried_same_native_seed",candidate,native);
    capture.immutable();input.check();
    const bool pass=cold.pass && carried.pass && propagated.pass;
    emit(@{@"kind":@"actual_capture_summary",@"qualification_pass":@(pass),
        @"native_recorded_capture_exact":@YES,@"cold_quality_pass":@(cold.pass),
        @"carried_native_seed_quality_pass":@(carried.pass),
        @"carried_candidate_seed_quality_pass":@(propagated.pass),
        @"manifest_sha256":ns(capture.manifestHash),@"v6_metallib_sha256":ns(V6LibrarySHA),
        @"canaries_pass":@YES,@"loaded_capture_immutable_sha256_pass":@YES,
        @"whole_model_qualified":@NO,@"selector_is_whole_history_certificate":@NO,
        @"timing_collected_even_if_numerical_gate_fails":@YES});
    return pass;
}
}

int main(int argc,char **argv) {
    @autoreleasepool {
        try {
            if (argc==1) {
                std::cout<<"Usage: metal-actual-capture --cpu-contract MANIFEST | --actual-capture MANIFEST [--library SEALED_V6_METALLIB]\n";
                return 0;
            }
            std::string mode,manifest,library=std::filesystem::absolute("build/gdn-nax-chunks-sep21-v6/gdn-wy.metallib").string();
            for (int i=1;i<argc;++i) {
                const std::string arg=argv[i];
                if (arg=="--cpu-contract" || arg=="--actual-capture") {
                    require(mode.empty(),"choose one actual capture mode");
                    require(++i<argc,"capture mode needs manifest path");mode=arg;manifest=argv[i];
                } else if (arg=="--library") {
                    require(++i<argc,"--library needs path");library=argv[i];
                } else throw std::runtime_error("unknown actual capture argument: "+arg);
            }
            require(!mode.empty(),"missing actual capture mode");
            const ActualCapture capture(manifest,mode=="--actual-capture");
            if (mode=="--cpu-contract") return 0;
            return actualRun(capture,library)?0:2;
        } catch (const std::exception &e) {
            emit(@{@"kind":@"error",@"message":ns(e.what())});return 1;
        }
    }
}
