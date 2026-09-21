"""Create reviewable production candidates without writing runtime paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRIVATE = Path(__file__).resolve().parent
OUT = PRIVATE / "production_candidate"
OUT.mkdir(exist_ok=True)

def replace_checked(text, before, after):
    if before not in text:
        raise ValueError(f"missing transform anchor: {before[:100]}")
    return text.replace(before, after)

policy = (PRIVATE / "Policy.hpp").read_text().replace("private_idle_maintenance", "idle_maintenance").replace("SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY", "SPLASH_FLASH_IDLE_RESIDENCY").replace("private idle maintenance", "idle residency maintenance")
(OUT / "FlashIdleResidencyPolicy.hpp").write_text(policy)
scheduler = (PRIVATE / "Scheduler.hpp").read_text().replace("private_idle_maintenance", "idle_maintenance")
(OUT / "FlashIdleResidencyScheduler.hpp").write_text(scheduler)
maintenance = (PRIVATE / "Maintenance.hpp").read_text().replace('"Policy.hpp"', '"flash/FlashIdleResidencyPolicy.hpp"').replace("private_idle_maintenance", "idle_maintenance").replace("flash_private_idle_immutable_touch_v11", "flash_idle_immutable_touch_v1").replace("private idle", "idle residency")
(OUT / "FlashIdleResidencyMaintenance.hpp").write_text(maintenance)
(OUT / "FlashIdleResidencyMaintenance.cpp").write_text('#include "flash/FlashIdleResidencyMaintenance.hpp"\n')
shader = (PRIVATE / "maintenance.metal").read_text().replace("flash_private_idle_immutable_touch_v11", "flash_idle_immutable_touch_v1")
(OUT / "flash_idle_residency_maintenance.metal").write_text(shader)
worker = (PRIVATE / "FlashWorker.mm").read_text().replace('#include "Maintenance.hpp"', '#include "flash/FlashIdleResidencyMaintenance.hpp"').replace('#include "Scheduler.hpp"', '#include "flash/FlashIdleResidencyScheduler.hpp"').replace("private_idle_maintenance", "idle_maintenance").replace("SPLASH_FLASH_PRIVATE_IDLE_RESIDENCY", "SPLASH_FLASH_IDLE_RESIDENCY").replace("private_idle_residency_maintenance", "idle_residency_maintenance").replace("privateIdleMaintenance", "idleMaintenance").replace("private idle", "idle residency")
worker = replace_checked(worker,
    "uint32_t idleMaintenanceInterval = idle_maintenance::kIntervalMilliseconds)",
    "uint32_t idleMaintenanceInterval = idle_maintenance::kIntervalMilliseconds,\n         bool idleMaintenanceRequested = false, std::string idleMaintenanceFailureReason = {})")
worker = replace_checked(worker,
    "idleMaintenance_(std::move(idleMaintenance)),",
    "idleMaintenance_(std::move(idleMaintenance)),\n        idleMaintenanceRequested_(idleMaintenanceRequested),\n        idleMaintenanceFailureReason_(std::move(idleMaintenanceFailureReason)),")
worker = replace_checked(worker,
    "std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance_;",
    "std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance_;\n  const bool idleMaintenanceRequested_;\n  const std::string idleMaintenanceFailureReason_;")
worker = replace_checked(worker,
    'void Worker::tickIdleMaintenance() {\n  if (!idleMaintenance_) return;',
    'void Worker::tickIdleMaintenance() {\n  if (!idleMaintenance_) {\n    if (idleMaintenanceRequested_ && idleMaintenanceState_ == "disabled")\n      idleMaintenanceState_ = "unavailable_immutable_owner_union";\n    return;\n  }')
worker = replace_checked(worker,
    '<< R"(,"idle_residency_maintenance":{"requested":)" << (idleMaintenance_ ? "true" : "false")',
    '<< R"(,"idle_residency_maintenance":{"requested":)" << (idleMaintenanceRequested_ ? "true" : "false")\n      << R"(,"available":)" << (idleMaintenance_ ? "true" : "false")\n      << R"(,"failure_reason":)" << json::quote(idleMaintenanceFailureReason_)')
worker = replace_checked(worker,
    "std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance;\n      if (idleMaintenanceRequested)",
    "std::unique_ptr<idle_maintenance::Maintenance> idleMaintenance;\n      std::string idleMaintenanceFailureReason;\n      if (idleMaintenanceRequested)")
before = """        idle_maintenance::validateGeometry(weights.sourceIdentity(), weights.manifestFingerprint(),
            originalCount, originalBytes, original.size(), ownerBytes,
            descriptor.layers, descriptor.experts, descriptor.hiddenSize);
        const uint64_t planned = idle_maintenance::Maintenance::plannedBytes(backend);"""
after = """        idle_maintenance::validateGeometry(weights.sourceIdentity(), weights.manifestFingerprint(),
            originalCount, originalBytes, idle_maintenance::kOwnerCount, idle_maintenance::kOwnerBytes,
            descriptor.layers, descriptor.experts, descriptor.hiddenSize);
        if (original.size() != idle_maintenance::kOwnerCount || ownerBytes != idle_maintenance::kOwnerBytes) {
          idleMaintenanceFailureReason = "qualified model requires its complete verified saved operand and expert stores";
        } else {
        const uint64_t planned = idle_maintenance::Maintenance::plannedBytes(backend);"""
worker = replace_checked(worker, before, after)
worker = replace_checked(worker,
    "reservation->commit();\n      }\n      const uint64_t instance",
    "reservation->commit();\n        }\n      }\n      const uint64_t instance")
worker = replace_checked(worker,
    "std::move(idleMaintenance), idleMaintenanceInterval);",
    "std::move(idleMaintenance), idleMaintenanceInterval, idleMaintenanceRequested,\n                    std::move(idleMaintenanceFailureReason));")
(OUT / "FlashWorker.mm").write_text(worker)
print(OUT)
