"""Measure the shipped safe concat eagerly and under CUDA graph replay."""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from segomega.utils.cuda_cat import SafeCudaCat


def measure(fn, repetitions=40):
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    tick = time.perf_counter()
    start.record()
    for _ in range(repetitions):
        result = fn()
    end.record()
    end.synchronize()
    return dict(wall_ms=(time.perf_counter()-tick)*1000/repetitions,
                device_timeline_ms=start.elapsed_time(end)/repetitions)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='output JSON file')
    parser.add_argument('--large', action='store_true', help='include 4-GiB boundary cases; needs about 24 GiB free VRAM')
    args = parser.parse_args()
    torch.set_num_threads(4)
    rows = []
    for frames, layout in ([(1, 'cache')] + ([(226, 'cache'), (256, 'cache'), (782, 'register')] if args.large else [])):
        if layout == 'cache':
            a = torch.ones((1, frames, 2321, 1024), device='cuda')
            b = torch.full_like(a, 2)
            dim = -1
        else:
            a = torch.ones((1, frames, 17, 1024), device='cuda')
            b = torch.full((1, frames, 2304, 1024), 2., device='cuda')
            dim = 2

        def eager():
            with SafeCudaCat():
                return torch.cat([a, b], dim)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = eager()
        torch.cuda.current_stream().wait_stream(stream)
        del warm
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        before = torch.cuda.memory_allocated()
        tick = time.perf_counter()
        with torch.cuda.graph(graph, stream=stream):
            captured = eager()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter()-tick)*1000
        graph_allocation = torch.cuda.memory_allocated()-before

        def replay():
            graph.replay()
            return captured

        # Replay must consume changed input values, including beyond 4 GiB.
        a.fill_(3); b.fill_(7)
        replay()
        first, second = captured.split([a.shape[dim], b.shape[dim]], dim)
        assert int((first != 3).sum()) == 0
        assert int((second != 7).sum()) == 0
        assert captured[0, -1, -1, -1024:].cpu().eq(7).all()
        reference = eager()
        assert torch.equal(reference, captured)
        del reference, first, second

        timings = {'eager': [], 'graph': []}
        for trial in range(5):
            order = [('eager', eager), ('graph', replay)]
            if trial % 2:
                order.reverse()
            for name, fn in order:
                timings[name].append(measure(fn))
        medians = {name: {key: statistics.median(row[key] for row in values)
                          for key in ['wall_ms', 'device_timeline_ms']}
                   for name, values in timings.items()}
        row = dict(frames=frames, layout=layout, output_bytes=captured.numel()*captured.element_size(),
                   replay_changed_inputs_exact=True, eager_graph_bitwise_equal=True,
                   capture_ms=capture_ms, graph_allocation_bytes=graph_allocation,
                   medians=medians, timings=timings,
                   speedup=medians['eager']['wall_ms']/medians['graph']['wall_ms'])
        rows.append(row)
        print(json.dumps({key:value for key,value in row.items() if key != 'timings'}), flush=True)
        del captured, a, b, graph, stream
        gc.collect(); torch.cuda.empty_cache()

    result = dict(status='pass', torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                  scope='Resident GPU concat only; excludes input transfers, model attention, warmup and capture from steady-state timing.',
                  timing='Median of five alternating trials, each with 40 calls and one final synchronization; shared GPU.',
                  rows=rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
