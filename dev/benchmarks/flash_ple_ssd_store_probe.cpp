#include "flash/FlashPLESSDStore.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using splash::flash::FlashPLESSDStore;
int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: store-probe SPEC.txt");
    std::ifstream stream(argv[1]);
    uint64_t sourceCount, partCount, cacheBytes, scratchBytes, noCache, idCount;
    if (!(stream >> sourceCount >> partCount >> cacheBytes >> scratchBytes >> noCache >> idCount))
      throw std::runtime_error("invalid spec header");
    if (sourceCount > 128 || partCount > 128 || idCount > 65536 || noCache > 1)
      throw std::runtime_error("probe extent exceeds CPU qualification bounds");
    std::vector<FlashPLESSDStore::Source> sources(sourceCount);
    for (auto &source : sources) {
      std::string path;
      if (!(stream >> path >> source.byteCount)) throw std::runtime_error("invalid source record");
      source.path = path;
    }
    std::vector<FlashPLESSDStore::Part> parts(partCount);
    for (auto &part : parts) {
      if (!(stream >> part.rows >> part.weights.source >> part.weights.offset >> part.weights.rowStride
          >> part.scales.source >> part.scales.offset >> part.scales.rowStride
          >> part.biases.source >> part.biases.offset >> part.biases.rowStride))
        throw std::runtime_error("invalid part record");
    }
    std::vector<int64_t> ids(idCount);
    for (auto &id : ids) if (!(stream >> id)) throw std::runtime_error("invalid ID record");
    std::string trailing;
    if (stream >> trailing) throw std::runtime_error("unexpected trailing spec record");
    FlashPLESSDStore store(std::move(sources), std::move(parts),
                          {cacheBytes, scratchBytes, bool(noCache)});
    std::vector<uint8_t> output(ids.size() * splash::flash::kFlashPLESSDRowBytes);
    uint64_t checksum = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
      const auto start = std::chrono::steady_clock::now();
      store.lookupRows(ids, output);
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
      if (!pass) checksum = 1469598103934665603ULL;
      uint64_t passChecksum = 1469598103934665603ULL;
      for (auto byte : output) { passChecksum ^= byte; passChecksum *= 1099511628211ULL; }
      if (!pass) checksum = passChecksum;
      if (checksum != passChecksum) throw std::runtime_error("cold/warm output differs");
      const auto s = store.statistics();
      std::cout << "{\"pass\":" << pass << ",\"cpu_only\":true,\"rows\":" << ids.size()
                << ",\"elapsed_ns\":" << elapsed << ",\"checksum_fnv1a\":" << passChecksum
                << ",\"reads_cumulative\":" << s.readRequests
                << ",\"read_bytes_cumulative\":" << s.completedReadBytes
                << ",\"logical_miss_bytes_cumulative\":" << s.logicalMissBytes
                << ",\"cache_hits_cumulative\":" << s.cacheHitRows
                << ",\"cache_accounted_bytes\":" << s.cacheAccountedBytes
                << ",\"file_cache_bypass\":" << (s.fileCacheBypassEnabled ? "true" : "false")
                << "}\n";
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
