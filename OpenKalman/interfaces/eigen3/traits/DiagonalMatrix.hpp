/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Type traits as applied to Eigen::DiagonalMatrix.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_DIAGONALMATRIX_HPP
#define OPENKALMAN_EIGEN3_TRAITS_DIAGONALMATRIX_HPP

#include <type_traits>


namespace OpenKalman
{
  namespace interface
  {
    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct IndexTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr std::size_t max_indices = 2;

      template<std::size_t N, typename Arg>
      static constexpr auto get_index_descriptor(const Arg& arg)
      {
        if constexpr (SizeAtCompileTime == Eigen::Dynamic) return static_cast<std::size_t>(arg.rows());
        else return Dimensions<SizeAtCompileTime>{};
      }

      template<Likelihood b>
      static constexpr bool is_one_by_one = SizeAtCompileTime == 1 or (SizeAtCompileTime == Eigen::Dynamic and b == Likelihood::maybe);

      template<Likelihood b>
      static constexpr bool is_square = true;
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct Dependencies<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      static constexpr bool has_runtime_parameters = false;
      using type = std::tuple<
        typename Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>::DiagonalVectorType>;

      template<std::size_t i, typename Arg>
      static decltype(auto) get_nested_matrix(Arg&& arg)
      {
        static_assert(i == 0);
        return std::forward<Arg>(arg).diagonal();
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto d {make_self_contained(std::forward<Arg>(arg).diagonal())};
        return DiagonalMatrix<decltype(d)> {d};
      }
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct TriangularTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      template<TriangleType t, Likelihood b>
      static constexpr bool is_triangular = true;

      static constexpr bool is_triangular_adapter = false;

      static constexpr bool is_diagonal_adapter = true;
    };


    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct Conversions<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    {
      template<typename Arg>
      static decltype(auto)
      to_diagonal(Arg&& arg)
      {
        // In this case, arg will be one-by-one.
        if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1)
          throw std::logic_error {"Argument of to_diagonal must be 1-by-1"};

        using M = Eigen::Matrix<scalar_type_of_t<Arg>, 1, 1>;
        return M {std::forward<Arg>(arg).diagonal()};
      }


      template<typename Arg>
      static constexpr decltype(auto)
      diagonal_of(Arg&& arg)
      {
        auto d {make_self_contained<Arg>(std::forward<Arg>(arg).diagonal())};

        if constexpr (std::is_lvalue_reference_v<Arg> or not has_dynamic_dimensions<Arg> or SizeAtCompileTime == Eigen::Dynamic)
          return d;
        else
          return untyped_dense_writable_matrix_t<decltype(d), Scalar, static_cast<std::size_t>(SizeAtCompileTime), 1> {std::move(d)};
      }
    };

  } // namespace interface


  /**
   * \internal
   * \brief Matrix traits for Eigen::DiagonalMatrix.
   */
  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct MatrixTraits<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : MatrixTraits<Eigen::Matrix<Scalar, SizeAtCompileTime, SizeAtCompileTime>>
  {
  };

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_TRAITS_DIAGONALMATRIX_HPP
