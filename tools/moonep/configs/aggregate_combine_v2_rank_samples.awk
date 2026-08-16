$1 == "COMBINE_V2_SAMPLE" {
    rank = ""
    elapsed = ""
    for (field = 2; field <= NF; ++field) {
        split($field, item, "=")
        if (item[1] == "rank") rank = item[2]
        else if (item[1] == "elapsed_ms") elapsed = item[2]
    }
    count[rank]++
    total[rank] += elapsed
}

$1 == "COMBINE_V2_RANK_PERF" {
    rank = ""
    average = ""
    for (field = 2; field <= NF; ++field) {
        split($field, item, "=")
        if (item[1] == "rank") rank = item[2]
        else if (item[1] == "avg_ms") average = item[2]
    }
    perf_count[rank]++
    perf_average[rank] = average
}

END {
    for (rank = 0; rank < expected_ranks; ++rank) {
        if (perf_count[rank] == 1 && perf_average[rank] != "") {
            average = perf_average[rank]
        } else if (perf_count[rank] == 0 &&
                count[rank] == expected_iterations) {
            average = total[rank] / expected_iterations
        } else {
            printf "BAD rank=%d rank_perf=%d samples=%d\n", rank, \
                perf_count[rank], count[rank]
            invalid = 1
            continue
        }
        printf "RANK_AVG rank=%d avg_ms=%.9f\n", rank, average
        sum += average
        if (rank == 0 || average > maximum) {
            maximum = average
            maximum_rank = rank
        }
    }
    if (invalid) exit 1

    mean = sum / expected_ranks
    data_bytes = bs * 16 * 3584 * 2
    printf "AGG avg_ms=%.9f avg_alg_bw_GBps=%.9f max_ms=%.9f max_rank=%d max_alg_bw_GBps=%.9f\n", \
        mean, data_bytes / mean / 1000000, maximum, maximum_rank, \
        data_bytes / maximum / 1000000
}
