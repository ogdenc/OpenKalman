/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Definitions relating to general functions
 *
 * \file
 * \brief Header file for all general functions
 */

#ifndef OPENKALMAN_FUNCTIONS_HPP
#define OPENKALMAN_FUNCTIONS_HPP

#include "nested_object.hpp"

#include "internal/make_library_wrapper.hpp"

#include "internal/make_fixed_size_adapter.hpp"
#include "internal/make_fixed_size_adapter_like.hpp"
#include "internal/make_fixed_square_adapter_like.hpp"

#include "internal/truncate_indices.hpp"
#include "get_component.hpp"
#include "set_component.hpp"

#include "internal/may_hold_components.hpp"
#include "fill_components.hpp"

#include "to_native_matrix.hpp"

#include "make_dense_object.hpp"
#include "make_dense_object_from.hpp"
#include "make_self_contained.hpp"
#include "make_constant.hpp"
#include "make_zero.hpp"
#include "make_identity_matrix_like.hpp"

#include "to_diagonal.hpp"
#include "diagonal_of.hpp"

#include "conjugate.hpp"
#include "transpose.hpp"
#include "adjoint.hpp"

#include "make_triangular_matrix.hpp"
#include "make_hermitian_matrix.hpp"

#include "internal/to_covariance_nestable.hpp"

#include "to_euclidean.hpp"
#include "from_euclidean.hpp"
#include "wrap_angles.hpp"

#include "make_matrix.hpp"

#include "n_ary_operation.hpp"
#include "randomize.hpp"

#include "internal/make_constant_matrix_reduction.hpp"
#include "internal/has_uniform_reduction_indices.hpp"
#include "reduce.hpp"
#include "average_reduce.hpp"

#include "internal/check_block_limits.hpp"
#include "get_block.hpp"
#include "set_block.hpp"
#include "get_chip.hpp"
#include "set_chip.hpp"
#include "internal/set_triangle.hpp"

#include "tile.hpp"
#include "concatenate.hpp"
#include "split.hpp"
#include "chipwise_operation.hpp"

#include "determinant.hpp"
#include "trace.hpp"
#include "sum.hpp"

#include "contract.hpp"
#include "contract_in_place.hpp"

#include "LQ_decomposition.hpp"
#include "QR_decomposition.hpp"

#include "cholesky_square.hpp"
#include "cholesky_factor.hpp"

#include "internal/make_writable_square_matrix.hpp"
#include "rank_update_hermitian.hpp"
#include "rank_update_triangular.hpp"

#include "solve.hpp"


#endif //OPENKALMAN_FUNCTIONS_HPP
