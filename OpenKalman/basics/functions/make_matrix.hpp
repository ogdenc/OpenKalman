/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_matrix.
 */

#ifndef OPENKALMAN_MAKE_MATRIX_HPP
#define OPENKALMAN_MAKE_MATRIX_HPP

namespace OpenKalman
{
  /**
   * \brief Make a Matrix object from a typed_matrix_nestable, specifying the row and column coefficients.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
   * \tparam M A typed_matrix_nestable with size matching RowCoefficients and ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, fixed_vector_space_descriptor ColumnCoefficients, typed_matrix_nestable M>
    requires (index_dimension_of_v<M, 0> == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of_v<M, 1> == dimension_size_of_v<ColumnCoefficients>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<RowCoefficients> and fixed_vector_space_descriptor<ColumnCoefficients> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of<M, 1>::value == dimension_size_of_v<ColumnCoefficients>), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    return Matrix<RowCoefficients, ColumnCoefficients, passable_t<M>>(std::forward<M>(m));
  }


  /**
   * \brief Make a Matrix object from a typed_matrix_nestable, specifying only the row coefficients.
   * \details The column coefficients are default Axis.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching RowCoefficients and ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, typed_matrix_nestable M>
  requires (index_dimension_of_v<M, 0> == dimension_size_of_v<RowCoefficients>)
#else
  template<typename RowCoefficients, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<RowCoefficients> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<RowCoefficients>), int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using ColumnCoefficients = Dimensions<index_dimension_of_v<M, 1>>;
    return Matrix<RowCoefficients, ColumnCoefficients, passable_t<M>>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Matrix object from a typed_matrix_nestable object, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_matrix(M&& m)
  {
    using RowCoeffs = Dimensions<index_dimension_of_v<M, 0>>;
    using ColCoeffs = Dimensions<index_dimension_of_v<M, 1>>;
    return make_matrix<RowCoeffs, ColCoeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Matrix object from a covariance object.
   * \tparam M A covariance object (i.e., Covariance, SquareRootCovariance).
   */
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto make_matrix(M&& arg)
  {
    using C = vector_space_descriptor_of_t<M, 0>;
    return make_matrix<C, C>(make_dense_writable_matrix_from(std::forward<M>(arg)));
  }


  /**
   * \overload
   * \brief Make a Matrix object from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto make_matrix(Arg&& arg)
  {
    using RowCoeffs = vector_space_descriptor_of_t<Arg, 0>;
    using ColCoeffs = vector_space_descriptor_of_t<Arg, 1>;
    if constexpr(euclidean_transformed<Arg>)
      return make_matrix<RowCoeffs, ColCoeffs>(nested_matrix(from_euclidean<RowCoeffs>(std::forward<Arg>(arg))));
    else
      return make_matrix<RowCoeffs, ColCoeffs>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a default, self-contained Matrix object.
   * \tparam RowCoefficients The coefficient types corresponding to the rows.
   * \tparam ColumnCoefficients The coefficient types corresponding to the columns.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor RowCoefficients, fixed_vector_space_descriptor ColumnCoefficients, typed_matrix_nestable M> requires
    (index_dimension_of_v<M, 0> == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of_v<M, 1> == dimension_size_of_v<ColumnCoefficients>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename M, std::enable_if_t<
    fixed_vector_space_descriptor<RowCoefficients> and fixed_vector_space_descriptor<ColumnCoefficients> and typed_matrix_nestable<M> and
    (index_dimension_of<M, 0>::value == dimension_size_of_v<RowCoefficients>) and
    (index_dimension_of<M, 1>::value == dimension_size_of_v<ColumnCoefficients>), int> = 0>
#endif
  inline auto make_matrix()
  {
    return Matrix<RowCoefficients, ColumnCoefficients, dense_writable_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained Matrix object with default Axis coefficients.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_matrix()
  {
    using RowCoeffs = Dimensions<index_dimension_of_v<M, 0>>;
    using ColCoeffs = Dimensions<index_dimension_of_v<M, 1>>;
    return make_matrix<RowCoeffs, ColCoeffs, M>();
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_MATRIX_HPP
