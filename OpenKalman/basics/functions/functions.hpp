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

#include "indexible_property_functions.hpp"

#include "element_functions.hpp"

#include "to_native_matrix.hpp"

#include "make_default_dense_writable_matrix_like.hpp"
#include "make_dense_writable_matrix_from.hpp"
#include "make_self_contained.hpp"
#include "make_constant_matrix_like.hpp"
#include "make_zero_matrix_like.hpp"
#include "make_identity_matrix_like.hpp"

#include "to_diagonal.hpp"
#include "diagonal_of.hpp"

#include "conjugate.hpp"
#include "transpose.hpp"
#include "adjoint.hpp"

#include "make_triangular_matrix.hpp"
#include "make_hermitian_matrix.hpp"

#include "to_covariance_nestable.hpp"

#include "to_euclidean.hpp"
#include "from_euclidean.hpp"
#include "wrap_angles.hpp"

#include "n_ary_operation.hpp"
#include "randomize.hpp"
#include "reduction_functions.hpp"
#include "block_functions.hpp"
#include "tile.hpp"
#include "concatenate.hpp"
#include "split.hpp"
#include "chipwise-operations.hpp"

#include "determinant.hpp"
#include "trace.hpp"
#include "sum.hpp"
#include "contract.hpp"
#include "contract_in_place.hpp"

#include "decomposition_functions.hpp"
#include "cholesky-decomposition.hpp"
#include "rank-update.hpp"
#include "solve.hpp"

#endif //OPENKALMAN_FUNCTIONS_HPP
