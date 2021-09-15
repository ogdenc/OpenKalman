/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Concepts as applied to native Eigen3 matrix classes.
 */

#ifndef OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
#define OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{

  /*
   * \internal
   * \brief Default matrix traits for any \ref eigen_native.
   * \tparam M The matrix.
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_native M> requires std::same_as<M, std::decay_t<M>>
  struct MatrixTraits<M>
#else
  template<typename M>
  struct MatrixTraits<M, std::enable_if_t<Eigen3::eigen_native<M> and std::is_same_v<M, std::decay_t<M>>>>
#endif
  {
    using Scalar = typename M::Scalar;


    /** \todo Currently, c_rows and c_cols are necessary to avoid a bug in GCC 10.1.0 when "rows" is used
     * as a default template parameter. Might be worth updating later.
     */
    static constexpr std::size_t c_rows()
    {
      if constexpr (Eigen::internal::traits<M>::RowsAtCompileTime == Eigen::Dynamic) return 0;
      else return Eigen::internal::traits<M>::RowsAtCompileTime;
    }


    static constexpr std::size_t c_cols()
    {
      if constexpr (Eigen::internal::traits<M>::ColsAtCompileTime == Eigen::Dynamic) return 0;
      else return Eigen::internal::traits<M>::ColsAtCompileTime;
    }


    static constexpr std::size_t rows = c_rows();


    static constexpr std::size_t columns = c_cols();

  private:

    // Identify the correct Eigen::Matrix based on template parameters and the traits of M.
    template<typename S, std::size_t r, std::size_t c>
    using Nat =
      Eigen::Matrix<S, r == 0 ? Eigen::Dynamic : (Eigen::Index) r, c == 0 ? Eigen::Dynamic : (Eigen::Index) c,
        (Eigen::internal::traits<M>::Flags & Eigen::RowMajorBit and (c != 1) ?
          Eigen::RowMajor : Eigen::ColMajor) | Eigen::AutoAlign,
        r == 0 ? Eigen::internal::traits<M>::MaxRowsAtCompileTime : (Eigen::Index) r,
        c == 0 ? Eigen::internal::traits<M>::MaxColsAtCompileTime : (Eigen::Index) c>;

  public:

    template<std::size_t r = c_rows(), std::size_t c = c_cols(), typename S = Scalar>
    using NativeMatrixFrom = Nat<S, r, c>;


    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = c_rows(), typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Nat<S, dim, dim>, storage_triangle>;


    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = c_rows(), typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Nat<S, dim, dim>, triangle_type>;


    template<std::size_t dim = c_rows(), typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<Nat<S, dim, 1>>;


    using SelfContainedFrom = NativeMatrixFrom<>;


    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, M>;


#ifdef __cpp_concepts
    template<Eigen3::eigen_native Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_native<Arg>, int> = 0>
#endif
    static decltype(auto) make(Arg&& arg) noexcept
    {
      return std::forward<Arg>(arg);
    }


    // Make matrix of size M from a list of coefficients in row-major order. Fixed matrix.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> Arg, std::convertible_to<Scalar> ... Args> requires
      (1 + sizeof...(Args) == rows * columns) and (not dynamic_rows<M>) and (not dynamic_columns<M>)
#else
    template<typename Arg, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...> and
      (1 + sizeof...(Args) == rows * columns) and (not dynamic_rows<M>) and (not dynamic_columns<M>), int> = 0>
#endif
    static auto make(const Arg arg, const Args ... args)
    {
      return ((Nat<Scalar, rows, columns> {} << arg), ... , args).finished();
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return Eigen3::ZeroMatrix<Scalar, rows, columns> {static_cast<std::size_t>(args)...};
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Eigen::Index> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, Eigen::Index> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      return Nat<Scalar, rows, rows>::Identity(static_cast<Eigen::Index>(args)..., static_cast<Eigen::Index>(args)...);
    }

  };


  /*
   * \internal
   * \brief Matrix traits for Eigen::SelfAdjointView.
   */
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>> : MatrixTraits<M>
  {
    using MatrixTraits<M>::rows;

    using typename MatrixTraits<M>::Scalar;

    static constexpr TriangleType storage_triangle = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;


    template<TriangleType storage_triangle = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = typename MatrixTraits<M>::template SelfAdjointMatrixFrom<storage_triangle, dim, S>;


    template<TriangleType triangle_type = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = typename MatrixTraits<M>::template TriangularMatrixFrom<triangle_type, dim, S>;


    using SelfContainedFrom = std::conditional_t<self_adjoint_matrix<M>, self_contained_t<M>, SelfAdjointMatrixFrom<>>;


#ifdef __cpp_concepts
    template<Eigen3::eigen_native Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_native<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::SelfAdjointView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };


  /*
   * \internal
   * \brief Matrix traits for Eigen::TriangularView.
   */
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::TriangularView<M, UpLo>> : MatrixTraits<M>
  {
    using MatrixTraits<M>::rows;

    using typename MatrixTraits<M>::Scalar;

    static constexpr TriangleType triangle_type = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;


    template<TriangleType storage_triangle = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = typename MatrixTraits<M>::template SelfAdjointMatrixFrom<storage_triangle, dim, S>;


    template<TriangleType triangle_type = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = typename MatrixTraits<M>::template TriangularMatrixFrom<triangle_type, dim, S>;


    using SelfContainedFrom = std::conditional_t<
      (lower_triangular_matrix<M> and (triangle_type == TriangleType::lower)) or
      (upper_triangular_matrix<M> and (triangle_type == TriangleType::upper)),
        self_contained_t<M>, TriangularMatrixFrom<>>;


#ifdef __cpp_concepts
    template<Eigen3::eigen_native Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_native<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::TriangularView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };


} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
