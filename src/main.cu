/**
 * SGEMM Optimization Benchmark
 *
 * 应用程序入口点，负责：
 * - 解析命令行参数
 * - 创建并运行 Benchmark 编排器
 *
 * 业务逻辑委托给：
 * - cli_parser.cuh: CLI 解析
 * - benchmark_runner.cuh: Benchmark 编排
 */

#include "benchmark_runner.cuh"
#include "cli_parser.cuh"

int main(int argc, char **argv) {
    BenchmarkConfig config;
    CliParser parser(argc, argv);

    ParseResult result = parser.parse(config);
    switch (result) {
    case ParseResult::HelpShown:
        return 0;
    case ParseResult::Error:
        return 1;
    case ParseResult::Success:
        break;
    }

    BenchmarkRunner runner(config);
    return runner.runAll() ? 0 : 1;
}
