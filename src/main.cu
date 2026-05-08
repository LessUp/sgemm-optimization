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

int main(int argc, char** argv) {
    BenchmarkConfig config;
    CliParser parser(argc, argv);

    int parse_result = parser.parse(config);
    if (parse_result == 2) {
        // 显示帮助后正常退出
        return 0;
    }
    if (parse_result != 0) {
        // 解析错误
        return 1;
    }

    BenchmarkRunner runner(config);
    runner.runAll();

    return 0;
}
