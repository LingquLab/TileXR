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

END {
    for (rank = 0; rank < expected_ranks; ++rank) {
        if (count[rank] != expected_iterations) {
            printf "BAD rank=%d samples=%d\n", rank, count[rank]
            invalid = 1
            continue
        }
        average = total[rank] / expected_iterations
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
