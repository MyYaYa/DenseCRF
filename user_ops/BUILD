load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

cc_library(
	name = "permutohedral",
	srcs = ["permutohedral.cc"],
	hdrs = ["permutohedral.h"],
)

cc_library(
	name = "densecrf_util",
	srcs = ["densecrf_util.cc"],
	hdrs = ["densecrf_util.h"],
)

cc_library(
	name = "densecrf_pairwise",
	srcs = ["densecrf_pairwise.cc"],
	hdrs = ["densecrf_pairwise.h"],
	deps = [
		":permutohedral",
		":densecrf_util",
	],
)


tf_custom_op_library(
	name = "densecrf.so",
	srcs = ["densecrf.cc"],
	deps = [":densecrf_pairwise"]
)
