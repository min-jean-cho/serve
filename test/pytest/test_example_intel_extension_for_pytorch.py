import json
import os

import requests
import test_utils
from intel_extension_for_pytorch.cpu.launch import CPUinfo
from test_handler import run_inference_using_url_with_data

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
snapshot_file_ipex = os.path.join(REPO_ROOT, "test/config_ipex.properties")
data_file_kitten = os.path.join(REPO_ROOT, "examples/image_classifier/kitten.jpg")
TS_LOG = "./logs/ts_log.log"

MANAGEMENT_API = "http://localhost:8081"
INFERENCE_API = "http://localhost:8080"

cpuinfo = CPUinfo()
NUM_CORES = cpuinfo.physical_core_nums()


def setup_module():
    test_utils.torchserve_cleanup()
    response = requests.get(
        "https://torchserve.pytorch.org/mar_files/resnet-18.mar", allow_redirects=True
    )
    open(test_utils.MODEL_STORE + "resnet-18.mar", "wb").write(response.content)


def setup_torchserve():
    if os.path.exists(TS_LOG):
        os.remove(TS_LOG)
    test_utils.start_torchserve(
        test_utils.MODEL_STORE, snapshot_file_ipex, gen_mar=False
    )


def get_worker_affinity(num_cores, num_workers, worker_idx):
    num_cores_per_worker = num_cores // num_workers
    start = worker_idx * num_cores_per_worker
    end = (worker_idx + 1) * num_cores_per_worker - 1
    curr_worker_cores = [i for i in range(start, end + 1)]
    affinity = "numactl -C {}-{}".format(str(start), str(end))
    affinity += " -m {}".format(
        ",".join(
            [str(numa_id) for numa_id in cpuinfo.numa_aware_check(curr_worker_cores)]
        )
    )
    return affinity


def run_inference_with_core_pinning():
    files = {
        "data": (data_file_kitten, open(data_file_kitten, "rb")),
    }
    response = run_inference_using_url_with_data(
        "http://localhost:8080/predictions/resnet-18", files
    )
    return response


def scale_workers_with_core_pinning(scaled_num_workers):
    params = (("min_worker", str(scaled_num_workers)),)
    requests.put("http://localhost:8081/models/resnet-18", params=params)
    response = requests.get("http://localhost:8081/models/resnet-18")
    return response


def test_single_worker_affinity():
    num_workers = 1
    worker_idx = 0
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            num_workers
        )
    )

    response = run_inference_with_core_pinning()
    assert (
        response.status_code == 200
    ), "single-worker inference with core pinning failed"

    launcher_available = (
        "CPU launcher is enabled but launcher is not available. Proceeding without launcher."
        not in open(TS_LOG).read()
    )
    if launcher_available:
        affinity = get_worker_affinity(NUM_CORES, num_workers, worker_idx)
        assert (
            affinity in open(TS_LOG).read()
        ), "workers are not correctly pinned to cores"


def test_multi_worker_affinity():
    num_workers = 4
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            num_workers
        )
    )

    response = run_inference_with_core_pinning()
    assert (
        response.status_code == 200
    ), "multi-worker inference with core pinning failed"

    launcher_available = (
        "CPU launcher is enabled but launcher is not available. Proceeding without launcher."
        not in open(TS_LOG).read()
    )
    if launcher_available:
        for worker_idx in range(num_workers):
            curr_worker_affinity = get_worker_affinity(
                NUM_CORES, num_workers, worker_idx
            )
            assert (
                curr_worker_affinity in open(TS_LOG).read()
            ), "workers are not correctly pinned to cores"


def test_worker_scale_up_affinity():
    initial_num_workers = 2
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            initial_num_workers
        )
    )

    scaled_up_num_workers = 4
    response = scale_workers_with_core_pinning(scaled_up_num_workers)
    resnet18_list = json.loads(response.content)
    assert (
        len(resnet18_list[0]["workers"]) == scaled_up_num_workers
    ), "workers failed to scale up with core pinning"

    response = run_inference_with_core_pinning()
    assert (
        response.status_code == 200
    ), "scaled up workers inference with core pinning failed"

    launcher_available = (
        "CPU launcher is enabled but launcher is not available. Proceeding without launcher."
        not in open(TS_LOG).read()
    )
    if launcher_available:
        for worker_idx in range(scaled_up_num_workers):
            curr_worker_affinity = get_worker_affinity(
                NUM_CORES, scaled_up_num_workers, worker_idx
            )
            assert (
                curr_worker_affinity in open(TS_LOG).read()
            ), "workers are not correctly pinned to cores"


def test_worker_scale_down_affinity():
    initial_num_workers = 4
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            initial_num_workers
        )
    )

    scaled_down_num_workers = 2
    response = scale_workers_with_core_pinning(scaled_down_num_workers)
    resnet18_list = json.loads(response.content)
    assert (
        len(resnet18_list[0]["workers"]) == scaled_down_num_workers
    ), "workers failed to scale down with core pinning"

    response = run_inference_with_core_pinning()
    assert (
        response.status_code == 200
    ), "scaled down workers inference with core pinning failed"

    launcher_available = (
        "CPU launcher is enabled but launcher is not available. Proceeding without launcher."
        not in open(TS_LOG).read()
    )
    if launcher_available:
        for worker_idx in range(scaled_down_num_workers):
            curr_worker_affinity = get_worker_affinity(
                NUM_CORES, scaled_down_num_workers, worker_idx
            )
            assert (
                curr_worker_affinity in open(TS_LOG).read()
            ), "workers are not correctly pinned to cores"
