# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

LOG_FILE = os.path.join("debug_logs", "debug_log_globalrank-0.log")


def reset_debug_log():
    if os.path.isfile(LOG_FILE):
        # delete all content
        with open(LOG_FILE, "w") as _:
            pass


def check_debug_log(msg):
    with open(LOG_FILE) as f:
        for line in f.readlines():
            if msg in line:
                return True
    return False
