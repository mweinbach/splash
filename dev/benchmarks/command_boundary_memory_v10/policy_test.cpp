#include "Policy.hpp"
#include "metal/MetalBackend.hpp"
#include <iostream>
#include <thread>
#include <vector>

using namespace splash::metal::private_boundary_memory;
int main() {
    uint64_t checks = 0;
    auto require = [&](bool ok) {
        ++checks;
        if (!ok) throw std::runtime_error("private boundary-memory CPU check failed");
    };
    require(!parseSwitch(nullptr));
    require(!parseSwitch("0"));
    require(parseSwitch("1"));
    for (const char *raw : {"", "00", "01", "2", "true", " 1", "1 "}) {
        bool rejected = false;
        try { (void)parseSwitch(raw); }
        catch (const std::invalid_argument &) { rejected = true; }
        require(rejected);
    }
    for (bool skip : {false, true}) {
        Counters counters;
        std::vector<std::thread> threads;
        for (size_t worker = 0; worker < 4; ++worker) {
            threads.emplace_back([&, worker] {
                for (size_t i = 0; i < 10000; ++i)
                    if (counters.query(skip, static_cast<Boundary>(worker)) != !skip)
                        std::terminate();
            });
        }
        for (auto &thread : threads) thread.join();
        for (size_t i = 0; i < static_cast<size_t>(Boundary::Count); ++i) {
            require(counters.queried[i].load() == (skip ? 0 : 10000));
            require(counters.skipped[i].load() == (skip ? 10000 : 0));
        }
    }
    require(sizeof(splash::metal::CommandTiming) == 200);
    std::cout << "{\"valid\":true,\"gpu_work\":false,\"checks\":" << checks << "}\n";
}
