"""
Comprehensive Parameter Registry for linux_performance breeder.

This registry contains commonly-tuned Linux performance parameters
for networking, memory, CPU, and filesystem optimization.

Generated: 2025-01-15
"""

PARAMETER_REGISTRY = {
    # =========================================================================
    # TCP/IP PARAMETERS (net.ipv4.*)
    # =========================================================================

    # TCP Buffer Management
    "net.ipv4.tcp_rmem": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP read buffer (min, default, max)",
    },
    "net.ipv4.tcp_wmem": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP write buffer (min, default, max)",
    },
    "net.ipv4.tcp_mem": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP memory pages (min, pressure, max)",
    },

    # TCP Connection Management
    "net.ipv4.tcp_fin_timeout": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP FIN timeout in seconds",
        "typical_range": [5, 120],
    },
    "net.ipv4.tcp_tw_reuse": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Reuse TIME_WAIT sockets",
        "available_values": [0, 1],
        "value_descriptions": {
            "0": "Disabled",
            "1": "Enabled"
        },
    },
    "net.ipv4.tcp_tw_recycle": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Recycle TIME_WAIT sockets (deprecated in newer kernels)",
        "available_values": [0, 1],
    },
    "net.ipv4.tcp_max_tw_buckets": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max TIME_WAIT sockets",
        "typical_range": [8000, 600000],
    },
    "net.ipv4.tcp_max_syn_backlog": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max pending SYN connections",
        "typical_range": [128, 8192],
    },

    # TCP Keepalive
    "net.ipv4.tcp_keepalive_time": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Keepalive time in seconds",
    },
    "net.ipv4.tcp_keepalive_intvl": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Keepalive interval in seconds",
    },
    "net.ipv4.tcp_keepalive_probes": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Keepalive probes before dropping",
    },

    # TCP Performance & Features
    "net.ipv4.tcp_window_scaling": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP window scaling (RFC 7323)",
        "available_values": [0, 1],
    },
    "net.ipv4.tcp_sack": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP selective acknowledgments",
        "available_values": [0, 1],
    },
    "net.ipv4.tcp_timestamps": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP timestamps (PAWS)",
        "available_values": [0, 1],
    },
    "net.ipv4.tcp_fastopen": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP Fast Open (TFO)",
        "typical_range": [0, 1024],
    },
    "net.ipv4.tcp_slow_start_after_idle": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Seconds of idle before slow start",
        "typical_range": [0, 300],
    },
    "net.ipv4.tcp_mtu_probing": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "MTU probing",
        "typical_range": [0, 2],
    },
    "net.ipv4.tcp_frto": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Forward RTO (F-RTO)",
        "typical_range": [0, 2],
    },

    # TCP Congestion Control
    "net.ipv4.tcp_congestion_control": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP congestion algorithm",
        "available_values": ["reno", "cubic", "bbr", "bbr2", "htcp", "veno", "scalable"],
    },
    "net.ipv4.tcp_allowed_congestion_control": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Allowed congestion control algorithms",
    },
    "net.ipv4.tcp_available_congestion_control": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Available congestion control algorithms (read-only)",
    },

    # TCP Retransmission & Recovery
    "net.ipv4.tcp_retries1": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP retries before giving up",
        "typical_range": [3, 15],
    },
    "net.ipv4.tcp_retries2": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP retries on timeout",
        "typical_range": [5, 15],
    },
    "net.ipv4.tcp_reordering": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "TCP packet reordering threshold",
        "typical_range": [0, 300],
    },
    "net.ipv4.tcp_syn_retries": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "SYN retries",
        "typical_range": [1, 10],
    },
    "net.ipv4.tcp_synack_retries": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "SYN-ACK retries",
        "typical_range": [1, 10],
    },

    # UDP Parameters
    "net.core.rmem_default": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Default socket receive buffer",
    },
    "net.core.wmem_default": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Default socket send buffer",
    },
    "net.core.rmem_max": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max socket receive buffer",
    },
    "net.core.wmem_max": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max socket send buffer",
    },

    # Network Device
    "net.core.netdev_budget": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "NAPI polling budget",
        "typical_range": [10, 600],
    },
    "net.core.netdev_max_backlog": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Device backlog queue",
        "typical_range": [100, 10000],
    },
    "net.core.dev_weight": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Device processing weight",
        "typical_range": [10, 2000],
    },
    "net.core.netdev_budget_usecs": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "NAPI budget in microseconds",
        "typical_range": [1000, 10000],
    },
    "net.core.somaxconn": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max pending connections",
        "typical_range": [128, 65535],
    },
    "net.core.message_burst": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Message burst size",
    },
    "net.core.message_cost": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Message cost",
    },
    "net.core.optmem_max": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max option memory buffer",
    },

    # IP Layer
    "net.ipv4.ip_local_port_range": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Local port range (min max)",
    },
    "net.ipv4.ip_local_reserved_ports": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Reserved local ports",
    },
    "net.ipv4.ip_nonlocal_bind": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Bind to non-local IP",
        "typical_range": [0, 1],
    },
    "net.ipv4.ip_forward": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "IP forwarding",
        "typical_range": [0, 1],
    },
    "net.ipv4.conf.all.forwarding": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "All interfaces forwarding",
        "typical_range": [0, 1],
    },
    "net.ipv4.conf.default.forwarding": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Default interface forwarding",
        "typical_range": [0, 1],
    },

    # ICMP
    "net.ipv4.icmp_echo_ignore_all": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Ignore all ICMP echoes",
        "typical_range": [0, 1],
    },
    "net.ipv4.icmp_echo_ignore_broadcasts": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Ignore broadcast ICMP echoes",
        "typical_range": [0, 1],
    },
    "net.ipv4.icmp_ratelimit": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "ICMP rate limiting",
        "typical_range": [0, 1000],
    },
    "net.ipv4.icmp_msgs_per_sec": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "ICMP messages per second rate limit",
    },
    "net.ipv4.icmp_msgs_burst": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "ICMP messages burst rate limit",
    },

    # TCP Options
    "net.ipv4.tcp_low_latency": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Low latency mode (deprecated)",
        "typical_range": [0, 1],
    },
    "net.ipv4.tcp_high_performance": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "High performance mode (deprecated)",
        "typical_range": [0, 1],
    },
    "net.ipv4.tcp_thin_linear_timeouts": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Thin linear timeouts",
        "typical_range": [0, 1],
    },
    "net.ipv4.tcp_limit_output_bytes": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Limit output bytes per flow",
        "typical_range": [0, 1073741824],
    },
    "net.ipv4.tcp_challenge_ack_limit": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Challenge ACK limit",
        "typical_range": [0, 1000],
    },

    # =========================================================================
    # MEMORY/VIRTUAL MEMORY (vm.*)
    # =========================================================================

    # Swap Management
    "vm.swappiness": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Swap tendency (0=avoid, 100=aggressive)",
        "typical_range": [0, 100],
    },
    "vm.vfs_cache_pressure": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Cache reclaim tendency (0=keep, 1000=aggressive)",
        "typical_range": [0, 10000],
    },

    # Dirty Page Management
    "vm.dirty_ratio": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Dirty pages % before writeback",
        "typical_range": [5, 40],
    },
    "vm.dirty_background_ratio": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Dirty pages % for background writeback",
        "typical_range": [1, 20],
    },
    "vm.dirty_bytes": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Dirty pages threshold (bytes, overrides ratio)",
        "typical_range": [0, 2147483648],
    },
    "vm.dirty_background_bytes": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Background writeback threshold (bytes)",
        "typical_range": [0, 2147483648],
    },
    "vm.dirty_expire_centisecs": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Dirty data expiry (centiseconds)",
        "typical_range": [100, 30000],
    },
    "vm.dirty_writeback_centisecs": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Writeback interval (centiseconds)",
        "typical_range": [100, 5000],
    },
    "vm.dirtytime_expire_seconds": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Dirty time expiry (seconds)",
        "typical_range": [1, 30000],
    },

    # Memory Overcommit
    "vm.overcommit_memory": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Overcommit handling",
        "available_values": [0, 1, 2],
        "value_descriptions": {
            "0": "Heuristic",
            "1": "Always",
            "2": "Strict accounting"
        },
    },
    "vm.overcommit_ratio": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Overcommit percentage (when overcommit_memory=2)",
        "typical_range": [0, 200],
    },

    # Page Cache & Reclamation
    "vm.min_free_kbytes": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Min free kilobytes",
        "typical_range": [65536, 262144],
    },
    "vm.pagecache": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Page cache % (deprecated in newer kernels)",
        "typical_range": [0, 100],
    },
    "vm.drop_caches": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Drop caches (1=pagecache, 2=slabs, 3=all)",
        "typical_range": [0, 3],
    },
    "vm.page_lock_unfairness": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Page lock fairness",
        "available_values": [0, 1],
    },
    "vm.panic_on_oom": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Panic on OOM",
        "typical_range": [0, 1],
    },
    "vm.oom_kill_allocating_task": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Kill allocating task on OOM",
        "typical_range": [0, 1],
    },
    "vm.oom_dump_tasks": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Dump tasks on OOM kill",
        "typical_range": [0, 1],
    },

    # Transparent Hugepages
    "vm.nr_hugepages": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Number of hugepages",
    },
    "vm.nr_overcommit_hugepages": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Overcommit hugepages",
    },
    "vm.hugetlb_shm_group": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Hugepage SHM group",
    },
    "vm.transparent_hugepage": {
        "type": "categorical",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Transparent hugepages",
        "available_values": ["always", "madvise", "never"],
    },

    # Memory Compaction
    "vm.extfrag_threshold": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Fragmentation threshold",
        "typical_range": [0, 1000],
    },
    "vm.compact_unevictable_allowed": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Allow compacting unevictable",
        "typical_range": [0, 1],
    },
    "vm.compact_memory": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Enable memory compaction",
        "typical_range": [0, 1],
    },

    # NUMA & Memory Access
    "vm.zone_reclaim_mode": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Zone reclaim mode",
        "typical_range": [0, 7],
    },
    "vm.min_unmapped_ratio": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Min unmapped ratio",
        "typical_range": [0, 100],
    },
    "vm.max_map_count": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max memory map areas",
        "typical_range": [65530, 262144],
    },

    # =========================================================================
    # FILESYSTEM (fs.*)
    # =========================================================================

    "fs.file-max": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max open files system-wide",
        "typical_range": [8192, 2097152],
    },
    "fs.file-nr": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Current file handles (read-only)",
    },
    "fs.inode-state": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Inode state (read-only)",
    },

    "fs.inotify.max_user_watches": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max inotify watches per user",
        "typical_range": [8192, 524288],
    },
    "fs.inotify.max_user_instances": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max inotify instances per user",
        "typical_range": [128, 8192],
    },
    "fs.inotify.max_queued_events": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max queued inotify events",
        "typical_range": [128, 16384],
    },

    "fs.aio-max-nr": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max AIO events",
        "typical_range": [65536, 2097152],
    },
    "fs.aio-nr": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Current AIO events (read-only)",
    },

    "fs.epoll.max_user_watches": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max epoll watches per user",
        "typical_range": [4096, 131072],
    },

    "fs.pipe-max-size": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max pipe size in bytes",
        "typical_range": [4096, 1048576],
    },
    "fs.pipe-user-pages-hard": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Hard pipe user pages limit",
        "typical_range": [0, 16384],
    },
    "fs.pipe-user-pages-soft": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Soft pipe user pages limit",
        "typical_range": [0, 16384],
    },

    "fs.protected_regular": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Protected regular files",
        "typical_range": [0, 1],
    },
    "fs.protected_fifos": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Protected FIFOs",
        "typical_range": [0, 1],
    },
    "fs.suid_dumpable": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "SUID dumpable",
        "typical_range": [0, 2],
    },

    # =========================================================================
    # KERNEL PARAMETERS (kernel.*)
    # =========================================================================

    "kernel.pid_max": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max PID value",
        "typical_range": [32768, 4194304],
    },
    "kernel.threads-max": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max threads",
        "typical_range": [2048, 524288],
    },
    "kernel.randomize_va_space": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "ASLR randomization",
        "typical_range": [0, 2],
    },
    "kernel.numa_balancing": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "NUMA balancing",
        "typical_range": [0, 1],
    },
    "kernel.shmmax": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max shared memory segment",
        "typical_range": [0, 68719476736],
    },
    "kernel.shmall": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max shared memory pages",
        "typical_range": [0, 4294967296],
    },
    "kernel.shm_rmid_forced": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Force SHM RMID",
        "typical_range": [0, 1],
    },
    "kernel.sem": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Semaphore limits (sem msl opm mnl)",
    },
    "kernel.msgmax": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max message queue size",
        "typical_range": [8192, 65536],
    },
    "kernel.msgmnb": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max message queue bytes",
        "typical_range": [16384, 65536],
    },
    "kernel.msgmni": {
        "type": "int",
        "category": "sysctl",
        "causes_downtime": False,
        "description": "Max message queue IDs",
        "typical_range": [16, 4096],
    },

    # =========================================================================
    # SYSFS PARAMETERS (sysfs category)
    # =========================================================================

    # CPU Frequency Scaling
    "cpu_governor": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "path": "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
        "description": "CPU frequency governor",
        "available_values": ["performance", "powersave", "ondemand", "conservative", "schedutil", "userspace"],
    },

    # Memory Management (sysfs)
    "transparent_hugepage": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "path": "/sys/kernel/mm/transparent_hugepage/enabled",
        "description": "Transparent hugepages",
        "available_values": ["always", "madvise", "never"],
    },
    "ksm_run": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "path": "/sys/kernel/mm/ksm/run",
        "description": "Kernel samepage merging",
        "typical_range": [0, 1],
    },
    "ksm_sleep_millisecs": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "path": "/sys/kernel/mm/ksm/sleep_millisecs",
        "description": "KSM sleep interval",
        "typical_range": [0, 10000],
    },
    "ksm_pages_to_scan": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "path": "/sys/kernel/mm/ksm/pages_to_scan",
        "description": "KSM pages to scan",
        "typical_range": [1, 1000000000],
    },

    # =========================================================================
    # QDISC (Queueing Discipline) PARAMETERS (sysfs/tc)
    # =========================================================================

    # FQ (Fair Queueing)
    "qdisc_fq_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ queue limit in packets",
        "typical_range": [10, 10000],
    },
    "qdisc_fq_flow_refill_delay": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ flow refill delay in ms",
        "typical_range": [0, 10000],
    },
    "qdisc_fq_low_rate_threshold": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ low rate threshold in packets/s",
        "typical_range": [1, 1000000],
    },
    "qdisc_fq_quantum": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ quantum in bytes",
        "typical_range": [1514, 30000],
    },
    "qdisc_fq_initial_quantum": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ initial quantum in bytes",
        "typical_range": [1514, 30000],
    },
    "qdisc_fq_pacing": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ pacing",
        "available_values": ["on", "off"],
    },
    "qdisc_fq_ce_threshold": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ CE threshold in bytes",
        "typical_range": [0, 1048576],
    },

    # FQ_CODEL (Fair Queueing with Controlled Delay)
    "qdisc_fq_codel_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL queue limit in packets",
        "typical_range": [10, 10240],
    },
    "qdisc_fq_codel_flows": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL number of flow buckets",
        "typical_range": [1, 65536],
    },
    "qdisc_fq_codel_quantum": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL quantum in bytes",
        "typical_range": [1514, 65535],
    },
    "qdisc_fq_codel_target": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL target queue delay in us",
        "typical_range": [100, 100000],
    },
    "qdisc_fq_codel_interval": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL interval in us",
        "typical_range": [1000, 1000000],
    },
    "qdisc_fq_codel_ce_threshold": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL CE threshold in us",
        "typical_range": [0, 100000],
    },
    "qdisc_fq_codel_ecn": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL ECN marking",
        "available_values": ["on", "off"],
    },
    "qdisc_fq_codel_drop_batch_size": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL drop batch size",
        "typical_range": [0, 256],
    },
    "qdisc_fq_codel_memory_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "FQ_CODEL memory limit in bytes",
        "typical_range": [0, 1073741824],
    },

    # CODEL (Controlled Delay)
    "qdisc_codel_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CODEL queue limit in packets",
        "typical_range": [10, 10240],
    },
    "qdisc_codel_target": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CODEL target queue delay in us",
        "typical_range": [100, 100000],
    },
    "qdisc_codel_interval": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CODEL interval in us",
        "typical_range": [1000, 1000000],
    },
    "qdisc_codel_ce_threshold": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CODEL CE threshold in us",
        "typical_range": [0, 100000],
    },
    "qdisc_codel_ecn": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CODEL ECN marking",
        "available_values": ["on", "off"],
    },

    # RED (Random Early Detection)
    "qdisc_red_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED queue limit in packets",
        "typical_range": [10, 10000],
    },
    "qdisc_red_min": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED minimum threshold in packets",
        "typical_range": [0, 5000],
    },
    "qdisc_red_max": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED maximum threshold in packets",
        "typical_range": [1, 10000],
    },
    "qdisc_red_avpkt": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED average packet size in bytes",
        "typical_range": [512, 1500],
    },
    "qdisc_red_burst": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED burst in packets",
        "typical_range": [1, 500],
    },
    "qdisc_red_probability": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED maximum marking probability",
        "typical_range": [0.0, 1.0],
    },
    "qdisc_red_ecn": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED ECN marking",
        "available_values": ["on", "off"],
    },
    "qdisc_red_harddrop": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED hard drop",
        "available_values": ["on", "off"],
    },
    "qdisc_red_adaptive": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "RED adaptive mode",
        "available_values": ["on", "off"],
    },

    # HTB (Hierarchy Token Bucket)
    "qdisc_htb_rate": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "HTB rate in kbps",
        "typical_range": [1, 10000000],
    },
    "qdisc_htb_ceil": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "HTB ceiling rate in kbps",
        "typical_range": [1, 10000000],
    },
    "qdisc_htb_burst": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "HTB burst in bytes",
        "typical_range": [0, 1048576],
    },
    "qdisc_htb_cburst": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "HTB ceiling burst in bytes",
        "typical_range": [0, 1048576],
    },
    "qdisc_htb_quantum": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "HTB quantum in bytes",
        "typical_range": [1514, 60000],
    },
    "qdisc_htb_oversubscribe": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "HTB oversubscribe mode",
        "available_values": ["on", "off"],
    },

    # TBF (Token Bucket Filter)
    "qdisc_tbf_rate": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "TBF rate in kbps",
        "typical_range": [1, 10000000],
    },
    "qdisc_tbf_burst": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "TBF burst in bytes",
        "typical_range": [0, 10485760],
    },
    "qdisc_tbf_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "TBF queue limit in bytes",
        "typical_range": [0, 104857600],
    },
    "qdisc_tbf_peakrate": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "TBF peak rate in kbps",
        "typical_range": [1, 10000000],
    },
    "qdisc_tbf_mtu": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "TBF MTU in bytes",
        "typical_range": [512, 65536],
    },

    # SFQ (Stochastic Fairness Queueing)
    "qdisc_sfq_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "SFQ queue limit in packets",
        "typical_range": [1, 65535],
    },
    "qdisc_sfq_perturb": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "SFQ perturb interval in seconds",
        "typical_range": [0, 600],
    },
    "qdisc_sfq_quantum": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "SFQ quantum in bytes",
        "typical_range": [1514, 30000],
    },
    "qdisc_sfq_depth": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "SFQ depth",
        "typical_range": [1, 128],
    },
    "qdisc_sfq_headdrop": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "SFQ head drop",
        "available_values": ["on", "off"],
    },
    "qdisc_sfq_redflowlimit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "SFQ RED flow limit in bytes",
        "typical_range": [0, 1048576],
    },

    # PFIFO (Priority FIFO)
    "qdisc_pfifo_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "PFIFO queue limit in packets",
        "typical_range": [1, 65535],
    },

    # BFIFO (Byte FIFO)
    "qdisc_bfifo_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "BFIFO queue limit in bytes",
        "typical_range": [0, 104857600],
    },

    # PRIO (Priority Scheduler)
    "qdisc_prio_bands": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "PRIO number of bands",
        "typical_range": [1, 16],
    },

    # CHOKe (CHOose and Keep for responsive flows, CHOose and Kill for unresponsive flows)
    "qdisc_choke_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE queue limit in packets",
        "typical_range": [1, 65535],
    },
    "qdisc_choke_min": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE minimum threshold",
        "typical_range": [0, 32767],
    },
    "qdisc_choke_max": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE maximum threshold",
        "typical_range": [1, 65535],
    },
    "qdisc_choke_avpkt": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE average packet size",
        "typical_range": [512, 1500],
    },
    "qdisc_choke_burst": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE burst",
        "typical_range": [1, 500],
    },
    "qdisc_choke_probability": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE maximum marking probability",
        "typical_range": [0.0, 1.0],
    },
    "qdisc_choke_ecn": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "CHOKE ECN marking",
        "available_values": ["on", "off"],
    },

    # NETEM (Network Emulator - for testing)
    "qdisc_netem_delay": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM delay in ms",
        "typical_range": [0.0, 10000.0],
    },
    "qdisc_netem_jitter": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM jitter in ms",
        "typical_range": [0.0, 1000.0],
    },
    "qdisc_netem_loss": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM packet loss percentage",
        "typical_range": [0.0, 100.0],
    },
    "qdisc_netem_corrupt": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM corruption percentage",
        "typical_range": [0.0, 100.0],
    },
    "qdisc_netem_duplicate": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM duplication percentage",
        "typical_range": [0.0, 100.0],
    },
    "qdisc_netem_limit": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM queue limit in packets",
        "typical_range": [1, 100000],
    },
    "qdisc_netem_gap": {
        "type": "int",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM gap (every Nth packet)",
        "typical_range": [0, 10000],
    },
    "qdisc_netem_reorder": {
        "type": "float",
        "category": "sysfs",
        "causes_downtime": False,
        "description": "NETEM reordering percentage",
        "typical_range": [0.0, 100.0],
    },

    # Qdisc type selection
    "qdisc": {
        "type": "categorical",
        "category": "sysfs",
        "causes_downtime": True,  # Changing queue discipline can disrupt traffic
        "path": "/sys/class/net/*/queue/disc",
        "description": "Queue discipline type",
        "available_values": [
            "fq", "fq_codel", "codel",
            "pfifo_fast", "pfifo", "bfifo",
            "htb", "tbf", "sfq", "prio",
            "red", "choke", "netem", "noqueue"
        ],
    },

    # =========================================================================
    # CPUFREQ PARAMETERS (cpufreq category)
    # =========================================================================

    "governor": {
        "type": "categorical",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "CPU frequency governor",
        "available_values": ["performance", "powersave", "ondemand", "conservative", "schedutil"],
    },
    "min_freq_ghz": {
        "type": "float",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Min CPU frequency (GHz)",
        "typical_range": [0.8, 3.0],
    },
    "max_freq_ghz": {
        "type": "float",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Max CPU frequency (GHz)",
        "typical_range": [1.2, 5.0],
    },
    "scaling_min_freq": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Scaling min frequency (kHz)",
    },
    "scaling_max_freq": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Scaling max frequency (kHz)",
    },
    "scaling_cur_freq": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Current scaling frequency (kHz, read-only)",
    },
    "ondemand_up_threshold": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Ondemand governor up threshold",
        "typical_range": [1, 100],
    },
    "ondemand_sampling_down_factor": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Ondemand governor sampling down factor",
        "typical_range": [1, 100000],
    },
    "ondemand_ignore_nice_load": {
        "type": "categorical",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Ondemand governor ignore nice load",
        "available_values": [0, 1],
    },
    "ondemand_powersave_bias": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Ondemand governor powersave bias",
        "typical_range": [0, 1000],
    },
    "conservative_up_threshold": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Conservative governor up threshold",
        "typical_range": [1, 100],
    },
    "conservative_down_threshold": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Conservative governor down threshold",
        "typical_range": [1, 100],
    },
    "conservative_freq_step": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Conservative governor frequency step",
        "typical_range": [1, 100],
    },
    "schedutil_rate_limit_us": {
        "type": "int",
        "category": "cpufreq",
        "causes_downtime": False,
        "description": "Schedutil rate limit in microseconds",
        "typical_range": [0, 10000],
    },
}

# =============================================================================
# ETHTOOL PARAMETERS (per-interface network settings)
# =============================================================================

ETHTOOL_PARAMS = {
    # NOTE: All ethtool parameters can cause brief traffic disruption (packet loss, link flap)
    # Based on: https://vswitchzero.com/2017/09/26/vmxnet3-rx-ring-buffer-exhaustion-and-packet-loss/
    # "Modifying NIC driver settings may cause a brief traffic disruption"

    # Offload Features
    "tso": {
        "type": "categorical",
        "causes_downtime": True,  # Can cause brief packet loss
        "description": "TCP Segmentation Offload",
        "available_values": ["on", "off"],
    },
    "gro": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Generic Receive Offload",
        "available_values": ["on", "off"],
    },
    "lro": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Large Receive Offload",
        "available_values": ["on", "off"],
    },
    "gso": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Generic Segmentation Offload",
        "available_values": ["on", "off"],
    },
    "ufo": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "UDP Fragmentation Offload",
        "available_values": ["on", "off"],
    },
    "sg": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Scatter-gather I/O",
        "available_values": ["on", "off"],
    },

    # Ring Buffers
    "rx": {
        "type": "int",
        "causes_downtime": True,
        "description": "RX ring buffer size",
        "typical_range": [256, 8192],
    },
    "tx": {
        "type": "int",
        "causes_downtime": True,
        "description": "TX ring buffer size",
        "typical_range": [256, 8192],
    },
    "rx_ring": {
        "type": "int",
        "causes_downtime": True,
        "description": "RX ring buffer (alias)",
        "typical_range": [256, 8192],
    },
    "tx_ring": {
        "type": "int",
        "causes_downtime": True,
        "description": "TX ring buffer (alias)",
        "typical_range": [256, 8192],
    },

    # Flow Control
    "rx_pause": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "RX pause frames",
        "available_values": ["on", "off", "auto"],
    },
    "tx_pause": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "TX pause frames",
        "available_values": ["on", "off", "auto"],
    },
    "autoneg": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Auto-negotiation",
        "available_values": ["on", "off"],
    },

    # RSS / Hashing
    "rss": {
        "type": "int",
        "causes_downtime": True,
        "description": "RSS hash bits",
        "typical_range": [0, 64],
    },
    "rxhash": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "RX hashing",
        "available_values": ["on", "off"],
    },

    # NIC-specific
    "speed": {
        "type": "int",
        "causes_downtime": True,
        "description": "Link speed (Mbps)",
        "typical_range": [10, 100000],
    },
    "duplex": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Duplex mode",
        "available_values": ["half", "full", "auto"],
    },
    "wol": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "Wake-on-LAN",
        "available_values": ["pumbag", "d", "g", "s", "a", "b"],
    },

    # Additional offload features
    "rxvlan": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "RX VLAN offload",
        "available_values": ["on", "off"],
    },
    "txvlan": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "TX VLAN offload",
        "available_values": ["on", "off"],
    },
    "ntuple": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "NTUPLE filters",
        "available_values": ["on", "off"],
    },
    "rxhash": {
        "type": "categorical",
        "causes_downtime": True,
        "description": "RX hashing",
        "available_values": ["on", "off"],
    },
}
