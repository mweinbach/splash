#include "Policy.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
using namespace splash::metal::private_completed_retention;
int main() {
 unsigned checks=0;
 assert(!parseSwitch(nullptr)); ++checks;
 assert(!parseSwitch("0")); ++checks;
 assert(parseSwitch("1")); ++checks;
 for (const char *s : {"", "2", "true", "01", " 1", "1 "}) {
  bool threw=false; try { (void)parseSwitch(s); } catch(const std::invalid_argument &) { threw=true; }
  assert(threw); ++checks;
 }
 Slots<std::shared_ptr<int>> slots;
 std::vector<std::weak_ptr<int>> owners;
 for (int i=0; i<10000; ++i) {
  auto item=std::make_shared<int>(i); owners.push_back(item);
  auto old=slots.publish(item); old.reset(); item.reset();
  for (int j=0;j<=i;++j) { assert(owners[j].expired()==(j<i-1)); ++checks; }
 }
 auto retired=slots.close();
 assert(slots.closed); ++checks;
 assert(!slots.held[0]&&!slots.held[1]); ++checks;
 retired={};
 for (auto &w:owners) { assert(w.expired()); ++checks; }
 auto fresh=std::make_shared<int>(42); auto returned=slots.publish(fresh);
 assert(returned==fresh&&!slots.held[0]&&!slots.held[1]); ++checks;
 struct Store { std::mutex mutex; Slots<std::shared_ptr<int>> slots; };
 auto store=std::make_shared<Store>(); std::weak_ptr<Store> weak=store;
 std::thread writer([weak] { for(int i=0;i<10000;++i) if(auto s=weak.lock()) {
  std::shared_ptr<int> dropped;
  { std::lock_guard lock(s->mutex); dropped=s->slots.publish(std::make_shared<int>(i)); }
 } });
 std::thread stopper([weak] { for(int i=0;i<100;++i) if(auto s=weak.lock()) {
  std::array<std::shared_ptr<int>,2> dropped;
  { std::lock_guard lock(s->mutex); dropped=s->slots.close(); }
 } });
 writer.join(); stopper.join();
 {std::lock_guard lock(store->mutex);assert(store->slots.closed&&!store->slots.held[0]&&!store->slots.held[1]);++checks;}
 store.reset(); assert(weak.expired()); ++checks;
 std::cout << "{\"valid\":true,\"checks\":" << checks << "}\n";
}
