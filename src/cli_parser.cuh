#pragma once

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Benchmark 配置
// ============================================================================

/**
 * Benchmark 运行配置
 */
struct BenchmarkConfig {
    int warmup_runs = 5;
    int benchmark_runs = 20;
    std::vector<std::tuple<int, int, int>> dimensions;

    // 默认测试用例
    static const std::vector<std::tuple<int, int, int>> DEFAULT_CASES;

    bool empty() const { return dimensions.empty(); }
    void addCase(int M, int K, int N) { dimensions.emplace_back(M, K, N); }
    void setDefault() { dimensions = DEFAULT_CASES; }
};

const std::vector<std::tuple<int, int, int>> BenchmarkConfig::DEFAULT_CASES = {
    {512, 512, 512},
    {1024, 1024, 1024},
    {256, 384, 640},
    {511, 513, 1025},
};

// ============================================================================
// CLI 解析结果
// ============================================================================

/**
 * CLI 解析结果枚举
 *
 * 自文档化的返回类型，替代 int 返回码。
 */
enum class ParseResult {
    Success,    // 解析成功，继续执行
    Error,      // 解析错误，返回错误码
    HelpShown   // 显示帮助后正常退出
};

// ============================================================================
// CLI 解析器
// ============================================================================

namespace detail {

// 安全的字符串到整数转换
inline bool safeStrToInt(const char *str, int *result, const char *argName) {
    if (str == nullptr || str[0] == '\0') {
        fprintf(stderr, "Error: %s requires a valid number\n", argName);
        return false;
    }

    char *endptr;
    errno = 0;
    long val = strtol(str, &endptr, 10);

    if (errno == ERANGE || val > INT_MAX || val < INT_MIN) {
        fprintf(stderr, "Error: %s value '%s' is out of range\n", argName, str);
        return false;
    }
    if (*endptr != '\0') {
        fprintf(stderr, "Error: Invalid %s value '%s' (not a valid integer)\n", argName, str);
        return false;
    }

    *result = static_cast<int>(val);
    return true;
}

} // namespace detail

/**
 * 命令行参数解析器
 *
 * 支持：
 * - -s, --size SIZE: 单个正方形矩阵
 * - --dims M K N: 显式维度
 * - -a, --all: 默认测试集
 * - --warmup N: 预热次数
 * - --benchmark N: 计时次数
 * - -h, --help: 帮助信息
 */
class CliParser {
  public:
    CliParser(int argc, char **argv) : argc_(argc), argv_(argv) {}

    /**
     * 解析命令行参数
     *
     * @param config 输出配置对象
     * @return ParseResult 枚举指示解析结果
     */
    ParseResult parse(BenchmarkConfig &config) {
        for (int i = 1; i < argc_; ++i) {
            std::string arg = argv_[i];

            if (arg == "-h" || arg == "--help") {
                printUsage(argv_[0]);
                return ParseResult::HelpShown;
            }

            if (arg == "-s" || arg == "--size") {
                if (!parseSizeArg(i, config))
                    return ParseResult::Error;
                continue;
            }

            if (arg == "--dims") {
                if (!parseDimsArg(i, config))
                    return ParseResult::Error;
                continue;
            }

            if (arg == "-a" || arg == "--all") {
                config.setDefault();
                continue;
            }

            if (arg == "--warmup") {
                if (!parseWarmupArg(i, config))
                    return ParseResult::Error;
                continue;
            }

            if (arg == "--benchmark") {
                if (!parseBenchmarkArg(i, config))
                    return ParseResult::Error;
                continue;
            }

            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            printUsage(argv_[0]);
            return ParseResult::Error;
        }

        // 默认添加 1024x1024x1024
        if (config.empty()) {
            config.addCase(1024, 1024, 1024);
        }

        return ParseResult::Success;
    }

    void printUsage(const char *program) const {
        printf("Usage: %s [options]\n", program);
        printf("\nOptions:\n");
        printf("  -s, --size SIZE          Benchmark one square SIZE x SIZE x SIZE case\n");
        printf("  --dims M K N            Benchmark one explicit M x K x N case\n");
        printf("  -a, --all               Run the default benchmark set\n");
        printf("  --warmup N              Number of warmup runs (default: 5)\n");
        printf("  --benchmark N           Number of benchmark runs (default: 20)\n");
        printf("  -h, --help              Show this help message\n");
        printf("\nDefault benchmark set includes:\n");
        printf("  - aligned square cases (512, 1024)\n");
        printf("  - one aligned non-square case (256 x 384 x 640)\n");
        printf("  - one unaligned edge case (511 x 513 x 1025)\n");
        printf("\nExamples:\n");
        printf("  %s -s 1024\n", program);
        printf("  %s --dims 256 384 640\n", program);
        printf("  %s -a --warmup 10 --benchmark 50\n", program);
    }

  private:
    bool parseSizeArg(int &i, BenchmarkConfig &config) {
        if (i + 1 >= argc_) {
            fprintf(stderr, "Error: -s requires a size argument\n");
            return false;
        }

        int size;
        if (!detail::safeStrToInt(argv_[++i], &size, "size")) {
            return false;
        }
        if (size <= 0) {
            fprintf(stderr, "Error: Size must be positive\n");
            return false;
        }

        config.addCase(size, size, size);
        return true;
    }

    bool parseDimsArg(int &i, BenchmarkConfig &config) {
        if (i + 3 >= argc_) {
            fprintf(stderr, "Error: --dims requires M K N arguments\n");
            return false;
        }

        int M, K, N;
        if (!detail::safeStrToInt(argv_[++i], &M, "M dimension") ||
            !detail::safeStrToInt(argv_[++i], &K, "K dimension") ||
            !detail::safeStrToInt(argv_[++i], &N, "N dimension")) {
            return false;
        }
        if (M <= 0 || K <= 0 || N <= 0) {
            fprintf(stderr, "Error: Dimensions must be positive\n");
            return false;
        }

        config.addCase(M, K, N);
        return true;
    }

    bool parseWarmupArg(int &i, BenchmarkConfig &config) {
        if (i + 1 >= argc_) {
            fprintf(stderr, "Error: --warmup requires a number argument\n");
            return false;
        }

        int warmup;
        if (!detail::safeStrToInt(argv_[++i], &warmup, "warmup")) {
            return false;
        }
        if (warmup < 0) {
            fprintf(stderr, "Error: Warmup runs must be non-negative\n");
            return false;
        }

        config.warmup_runs = warmup;
        return true;
    }

    bool parseBenchmarkArg(int &i, BenchmarkConfig &config) {
        if (i + 1 >= argc_) {
            fprintf(stderr, "Error: --benchmark requires a number argument\n");
            return false;
        }

        int bench;
        if (!detail::safeStrToInt(argv_[++i], &bench, "benchmark")) {
            return false;
        }
        if (bench <= 0) {
            fprintf(stderr, "Error: Benchmark runs must be positive\n");
            return false;
        }

        config.benchmark_runs = bench;
        return true;
    }

    int argc_;
    char **argv_;
};
