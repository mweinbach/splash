// Root-only explicit payload/GPU mode; sealed local component remains unchanged.
#define main sealed_tile_component_main
#include "../gdn_tile_chunk_sep21/metal_oracle.mm"
#undef main

namespace {
constexpr const char *TileLibrarySHA="1f71721efbd672d98828092997bcfd65aa89ec7d78a4086f9f627526dda9457e";
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
    Inputs input(gpu,gdn_chunk_cpu::continuation_fixture(2048));
    for (auto [key,buffer]:std::array<std::pair<const char *,Guarded *>,3>{{
        {"mixed",&input.mixed},{"decay",&input.decay},{"beta",&input.beta}}}) {
        NSData *tensor=capture.tensor(key);
        require(tensor.length==buffer->bytes,"actual Inputs extent");
        std::memcpy(buffer->data(),tensor.bytes,tensor.length);buffer->freeze();
    }
    TileChunkParams p{{2048,1,16,48,128,128,4,1e-6f,
        uint64_t(3*Mixed*2),uint64_t(State*4)},0,0};
    std::memcpy(input.params.data(),&p,sizeof(p));input.params.freeze();input.check();
    return input;
}

Inputs actualSlice(GPU &gpu,const Inputs &source,size_t begin,size_t count) {
    require(begin+count<=source.rows && count && count<=Time,"actual slice range");
    Inputs chunk(gpu,gdn_chunk_cpu::continuation_fixture(count));
    for (auto [target,input,rowBytes]:std::array<std::tuple<Guarded *,const Guarded *,size_t>,3>{{
        {&chunk.mixed,&source.mixed,Mixed*2},{&chunk.decay,&source.decay,Heads*4},
        {&chunk.beta,&source.beta,Heads*2}}}) {
        require(target->bytes==count*rowBytes,"actual slice extent");
        std::memcpy(target->data(),static_cast<const unsigned char *>(input->data())+begin*rowBytes,
            target->bytes);target->freeze();
    }
    chunk.check();return chunk;
}

NSDictionary *actualDecisionMetrics(const Results &r) {
    // Reuse the sealed oracle's validation/count helper. Preserve all actual
    // words by their digest; expose compact per-(chunk,value-tile) counts.
    NSMutableDictionary *metrics=[decisionMetrics(r) mutableCopy];
    [metrics removeObjectForKey:@"decisions"];
    NSMutableArray *groups=[NSMutableArray array];
    size_t replayRows=0,rangeHeads=0;
    for (size_t c=0;c<r.chunks;++c) {
        for (size_t h=0;h<Heads;++h) {
            const auto word=r.range.as<uint32_t>()[c*Heads+h];
            require(!(word&~0xfu),"undefined/unwritten static chunk range bits");
            rangeHeads+=word!=0;
        }
        for (size_t tile=0;tile<Tiles;++tile) {
            size_t native=0,safe=0,range=0,cancel=0,nonfinite=0,norm=0,forced=0;
            for (size_t h=0;h<Heads;++h) {
                const auto word=r.decisions.as<uint32_t>()[(c*Heads+h)*Tiles+tile];
                native+=(word&NativeBit)!=0;safe+=(word&NativeBit)==0;
                range+=(word&1)!=0;cancel+=(word&2)!=0;nonfinite+=(word&4)!=0;
                norm+=(word&8)!=0;forced+=(word&ForcedBit)!=0;
            }
            replayRows+=native*std::min(Time,r.rows-c*Time);
            [groups addObject:@{@"chunk":@(c),@"value_tile":@(tile),@"heads":@(Heads),
                @"native_tiles":@(native),@"safe_tiles":@(safe),@"range_tiles":@(range),
                @"cancellation_tiles":@(cancel),@"nonfinite_tiles":@(nonfinite),
                @"norm_tiles":@(norm),@"forced_tiles":@(forced)}];
        }
    }
    metrics[@"per_chunk_value_tile"]=groups;
    metrics[@"static_range_chunk_heads"]=@(rangeHeads);
    metrics[@"native_tile_row_steps"]=@(replayRows);
    metrics[@"native_value_coordinate_row_steps"]=@(replayRows*Tile);
    metrics[@"decision_words_sha256"]=ns(digest(r.decisions.data(),r.decisions.bytes));
    metrics[@"selection_scope"]=@"independent T32 chunk/V32 value tile; no fullRows head replay";
    return metrics;
}

struct ActualBranchProof {
    size_t nativeTiles=0,state=0,output=0,timedState=0,timedOutput=0,timedDecisions=0;
    bool pass() const {return !state && !output && !timedState && !timedOutput && !timedDecisions;}
};
ActualBranchProof actualBranchProof(GPU &gpu,Inputs &input,Results &timed,
    const std::vector<float> &candidateSeed,const char *stage) {
    // Same full incoming-seed tape contract as sealed fullHeadProof; actual
    // slices are raw copies so no synthetic fixture encoding supplies data.
    Results probe(gpu,input.rows,false,true);probe.reset(candidateSeed);
    dispatch(gpu,input,probe,Run::Probe);input.check();probe.check();probe.incoming.freeze();
    ActualBranchProof proof;
    proof.timedState=differingBytes(probe.state.data(),timed.state.data(),timed.state.bytes);
    proof.timedOutput=differingBytes(probe.output.data(),timed.output.data(),timed.output.bytes);
    proof.timedDecisions=differingBytes(probe.decisions.data(),timed.decisions.data(),timed.decisions.bytes);
    for (size_t c=0;c<probe.chunks;++c) {
        const size_t begin=c*Time,count=std::min(Time,probe.rows-begin);
        std::vector<float> seed(probe.incoming.as<float>()+c*State,
            probe.incoming.as<float>()+(c+1)*State);
        Inputs chunk=actualSlice(gpu,input,begin,count);
        Results native(gpu,count);native.reset(seed);
        // Stronger actual branch control: original V16/Time16 native, not the
        // candidate's V8 helper. Every comparison uses identical hybrid seed.
        dispatch(gpu,chunk,native,Run::OriginalNative);chunk.check();native.check();
        require(!*native.diagnostics.as<uint32_t>(),"actual scoped native chunk diagnostics");
        const auto *next=c+1<probe.chunks?probe.incoming.as<float>()+(c+1)*State:probe.state.as<float>();
        for (size_t h=0;h<Heads;++h) for (size_t tile=0;tile<Tiles;++tile) {
            const auto word=probe.decisions.as<uint32_t>()[(c*Heads+h)*Tiles+tile];
            if (!(word&NativeBit)) continue;
            ++proof.nativeTiles;
            proof.state+=differingBytes(next+(h*V+tile*Tile)*K,
                native.state.as<float>()+(h*V+tile*Tile)*K,Tile*K*4);
            for (size_t t=0;t<count;++t)
                proof.output+=differingBytes(probe.output.as<uint16_t>()+(begin+t)*Out+h*V+tile*Tile,
                    native.output.as<uint16_t>()+t*Out+h*V+tile*Tile,Tile*2);
        }
    }
    probe.incoming.immutable();input.check();probe.check();timed.check();
    require(!*probe.diagnostics.as<uint32_t>(),"actual probe diagnostics");
    emit(@{@"kind":@"actual_tile_chunk_native_branch_proof",@"stage":ns(stage),
        @"pass":@(proof.pass()),@"native_tiles_checked":@(proof.nativeTiles),
        @"native_state_mismatch_bytes":@(proof.state),@"native_output_mismatch_bytes":@(proof.output),
        @"candidate_vs_probe_state_mismatch_bytes":@(proof.timedState),
        @"candidate_vs_probe_output_mismatch_bytes":@(proof.timedOutput),
        @"candidate_vs_probe_decision_mismatch_bytes":@(proof.timedDecisions),
        @"identical_actual_hybrid_incoming_seed":@YES,
        @"reference":@"original frozen V16/Time16 native on raw captured chunk from probe's identical incoming hybrid F32 state",
        @"full_sequence_bit_identity_claim":@NO,@"whole_history_accuracy_certificate":@NO,
        @"incoming_tape_bytes":@(probe.incoming.bytes),@"tape_in_timed_benchmark":@NO,
        @"canaries_pass":@YES,@"immutable_inputs_seeds_and_incoming_tape_pass":@YES});
    return proof;
}

struct ActualQuality {bool pass;std::vector<float> candidateCarry;};
ActualQuality actualQuality(GPU &gpu,Inputs &input,const std::vector<float> &candidateSeed,
    const std::vector<float> &nativeSeed,const char *stage,Results &candidate,Results &native) {
    candidate.reset(candidateSeed);native.reset(nativeSeed);
    dispatch(gpu,input,native,Run::OriginalNative);
    dispatch(gpu,input,candidate,Run::Candidate);
    input.check();candidate.check();native.check();
    std::vector<double> stateTruth(State),outputTruth(input.rows*Out);
    for (size_t i=0;i<State;++i) stateTruth[i]=native.state.as<float>()[i];
    for (size_t i=0;i<outputTruth.size();++i) outputTruth[i]=from_bf16(native.output.as<uint16_t>()[i]);
    const auto state=errors(State,stateTruth,1,[&](size_t i){return double(candidate.state.as<float>()[i]);});
    const auto output=errors(outputTruth.size(),outputTruth,1,[&](size_t i){return double(from_bf16(candidate.output.as<uint16_t>()[i]));});
    const auto proof=actualBranchProof(gpu,input,candidate,candidateSeed,stage);
    const bool identicalSeed=!differingBytes(candidateSeed.data(),nativeSeed.data(),State*4);
    const bool pass=state.pass() && bf16Pass(output) && proof.pass() &&
        !*candidate.diagnostics.as<uint32_t>() && !*native.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"actual_tile_chunk_quality",@"stage":ns(stage),@"qualification_pass":@(pass),
        @"state_f32_vs_original_native":state.json(),@"recurrence_bf16_vs_original_native":bf16Metrics(output),
        @"native_chunk_same_hybrid_seed_byte_exact":@(proof.pass()),
        @"whole_sequence_reference_incoming_seed_identical":@(identicalSeed),
        @"candidate_incoming_seed_sha256":ns(digest(candidateSeed.data(),State*4)),
        @"native_incoming_seed_sha256":ns(digest(nativeSeed.data(),State*4)),
        @"selection":actualDecisionMetrics(candidate),@"candidate_diagnostics":@(*candidate.diagnostics.as<uint32_t>()),
        @"native_diagnostics":@(*native.diagnostics.as<uint32_t>()),@"canaries_pass":@YES,
        @"immutable_inputs_and_reset_seeds_pass":@YES,@"unchanged_f32_and_bf16_quality_gates":@YES,
        @"full_sequence_bit_identity_claim":@NO,@"whole_history_accuracy_certificate":@NO,
        @"reference":@"original frozen V16/Time16 recurrence; candidate-own carry versus native-own carry in future trajectory screen"});
    return {pass,candidate.fullCarry()};
}

void actualBench(GPU &gpu,Inputs &input,const std::vector<float> &seed,const char *stage,
    Results &candidate,Results &native) {
    candidate.reset(seed);native.reset(seed);
    auto run=[&](bool useCandidate) {return dispatch(gpu,input,useCandidate?candidate:native,
        useCandidate?Run::Candidate:Run::OriginalNative,true);};
    double warmed=0;unsigned commands=0;
    while(warmed<.150 || commands<8) {
        warmed+=run(false).gpu;warmed+=run(true).gpu;commands+=2;
        require(commands<4096,"bounded actual tile GPU warmup");
    }
    emit(@{@"kind":@"actual_tile_chunk_bench_warmup",@"stage":ns(stage),@"gpu_ms":@(warmed*1000),
        @"commands":@(commands),@"minimum_gpu_ms":@150,@"cpu_tensor_access_during_warmup":@NO});
    std::vector<double> nativeTimes,candidateTimes,speedups,wallSpeedups;
    for(unsigned pair=0;pair<10;++pair) {
        const bool reverse=(pair%2)!=0;const std::array<bool,4> order{{reverse,!reverse,!reverse,reverse}};
        std::array<Timing,2>a{},b{};unsigned ai=0,bi=0;
        for(unsigned slot=0;slot<4;++slot) {
            const auto t=run(order[slot]);if(order[slot])b[bi++]=t;else a[ai++]=t;
            emit(@{@"kind":@"actual_tile_chunk_bench_sample",@"stage":ns(stage),@"pair":@(pair),@"slot":@(slot),
                @"order":reverse?@"BAAB":@"ABBA",@"run":order[slot]?@"component":@"original_native_v16_t16",
                @"gpu_ms":@(t.gpu*1000),@"wall_ms":@(t.wall*1000)});
        }
        const double at=(a[0].gpu+a[1].gpu)*.5,bt=(b[0].gpu+b[1].gpu)*.5;
        nativeTimes.push_back(at);candidateTimes.push_back(bt);speedups.push_back(at/bt);
        wallSpeedups.push_back((a[0].wall+a[1].wall)/(b[0].wall+b[1].wall));
    }
    input.check();candidate.check();native.check();
    require(!*candidate.diagnostics.as<uint32_t>() && !*native.diagnostics.as<uint32_t>(),"actual timed tile diagnostics");
    emit(@{@"kind":@"actual_tile_chunk_bench_summary",@"stage":ns(stage),@"rows":@2048,@"lanes":@1,
        @"matched_pairs":@10,@"abba_pairs":@5,@"baab_pairs":@5,
        @"original_native_gpu_ms_median":@(median(nativeTimes)*1000),
        @"candidate_gpu_ms_median":@(median(candidateTimes)*1000),
        @"paired_gpu_speedup_median":@(median(speedups)),@"paired_wall_speedup_median":@(median(wallSpeedups)),
        @"selection":actualDecisionMetrics(candidate),@"prepared_bytes":@(candidate.prepared.bytes),
        @"range_bytes":@(candidate.range.bytes),@"decision_bytes":@(candidate.decisions.bytes),
        @"preparation_native_fallback_and_gpu_reset_included":@YES,
        @"cpu_tensor_access_during_timing":@NO,@"probe_or_seed_tape_in_timing":@NO,
        @"canaries_pass":@YES,@"immutable_inputs_and_reset_seeds_pass":@YES,
        @"scope":@"actual captured layer0 T32/V32 GDN component; not whole model prefill throughput"});
}

bool actualRun(const ActualCapture &capture,const std::string &library) {
    NSError *error=nil;NSData *libraryData=[NSData dataWithContentsOfFile:ns(library) options:0 error:&error];
    require(libraryData!=nil && digest(libraryData.bytes,libraryData.length)==TileLibrarySHA,
        "unchanged sealed local component library SHA required");
    GPU gpu(library);Inputs input=actualInputs(gpu,capture);
    const auto initial=capture.state("initial_state");
    Results candidate(gpu,2048),native(gpu,2048);native.reset(initial);
    dispatch(gpu,input,native,Run::OriginalNative);input.check();native.check();
    const size_t stateMismatch=differingBytes(native.state.data(),capture.tensor("expected_state").bytes,State*4);
    const size_t outputMismatch=differingBytes(native.output.data(),capture.tensor("expected_recurrence").bytes,2048*Out*2);
    const bool baselineExact=!stateMismatch && !outputMismatch && !*native.diagnostics.as<uint32_t>();
    emit(@{@"kind":@"actual_tile_chunk_native_recorded_provenance",@"byte_exact":@(baselineExact),
        @"state_mismatch_bytes":@(stateMismatch),@"recurrence_mismatch_bytes":@(outputMismatch),
        @"native_diagnostics":@(*native.diagnostics.as<uint32_t>()),@"candidate_executed":@NO,
        @"canaries_pass":@YES,@"immutable_inputs_and_reset_seed_pass":@YES});
    require(baselineExact,"recorded actual cold original native provenance gate failed");
    const auto cold=actualQuality(gpu,input,initial,initial,"cold",candidate,native);
    const std::vector<uint16_t>coldOutput(candidate.output.as<uint16_t>(),candidate.output.as<uint16_t>()+2048*Out);
    const std::vector<uint32_t>coldDecisions(candidate.decisions.as<uint32_t>(),
        candidate.decisions.as<uint32_t>()+candidate.decisions.bytes/4);
    const auto nativeCarry=capture.state("expected_state");
    const auto carried=actualQuality(gpu,input,nativeCarry,nativeCarry,"carried_same_native_seed",candidate,native);
    const auto future=actualQuality(gpu,input,cold.candidateCarry,nativeCarry,
        "future_candidate_own_carry_vs_native_own_carry",candidate,native);
    candidate.reset(initial);*candidate.diagnostics.as<uint32_t>()=0x80000000u;
    dispatch(gpu,input,candidate,Run::Candidate);input.check();candidate.check();
    require(*candidate.diagnostics.as<uint32_t>()==0x80000000u,"sticky tile diagnostic lost or extra diagnostic");
    require(!differingBytes(candidate.state.data(),cold.candidateCarry.data(),State*4) &&
        !differingBytes(candidate.output.data(),coldOutput.data(),2048*Out*2) &&
        !differingBytes(candidate.decisions.data(),coldDecisions.data(),candidate.decisions.bytes),
        "sticky diagnostic repeat changed actual cold tile state/output/decisions");
    emit(@{@"kind":@"actual_tile_chunk_sticky_diagnostics",@"preserved":@YES,
        @"sticky_sentinel":@(0x80000000u),@"repeat_state_output_decisions_byte_exact":@YES,@"canaries_pass":@YES});
    actualBench(gpu,input,initial,"cold",candidate,native);
    actualBench(gpu,input,nativeCarry,"carried_same_native_seed",candidate,native);
    capture.immutable();input.check();
    const bool pass=cold.pass && carried.pass && future.pass;
    emit(@{@"kind":@"actual_tile_chunk_summary",@"qualification_pass":@(pass),
        @"native_recorded_capture_exact":@YES,@"cold_quality_pass":@(cold.pass),
        @"carried_native_seed_quality_pass":@(carried.pass),@"future_own_carry_quality_pass":@(future.pass),
        @"manifest_sha256":ns(capture.manifestHash),@"sealed_component_metallib_sha256":ns(TileLibrarySHA),
        @"loaded_capture_immutable_sha256_pass":@YES,@"canaries_pass":@YES,
        @"full_sequence_bit_identity_claim":@NO,@"whole_history_accuracy_certificate":@NO,
        @"whole_model_qualified":@NO,@"timing_collected_even_if_numerical_gate_fails":@YES});
    return pass;
}
}

int main(int argc,char **argv) {
    @autoreleasepool {try {
        if(argc==1) {
            std::cout<<"Usage: metal-actual-tile --cpu-contract MANIFEST | --actual-capture MANIFEST [--library SEALED_LOCAL_METALLIB]\n";
            return 0;
        }
        std::string mode,manifest,library=std::filesystem::absolute("build/gdn-tile-chunk-sep21/gdn-tile.metallib").string();
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--cpu-contract" || arg=="--actual-capture") {
                require(mode.empty(),"choose one actual tile capture mode");
                require(++i<argc,"capture mode needs manifest");mode=arg;manifest=argv[i];
            }else if(arg=="--library") {require(++i<argc,"--library needs path");library=argv[i];}
            else throw std::runtime_error("unknown actual tile argument: "+arg);
        }
        require(!mode.empty(),"missing actual tile capture mode");
        const ActualCapture capture(manifest,mode=="--actual-capture");
        if(mode=="--cpu-contract")return 0;
        return actualRun(capture,library)?0:2;
    }catch(const std::exception &e) {emit(@{@"kind":@"error",@"message":ns(e.what())});return 1;}}
}
