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
 * \brief Definitions for Eigen3::TriangularMatrix
 */

#ifndef OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP
#define OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType triangle_type> requires
    (has_dynamic_dimensions<NestedMatrix> or square_matrix<NestedMatrix>) and (triangle_type != TriangleType::none)
#else
  template<typename NestedMatrix, TriangleType triangle_type>
#endif
  struct TriangularMatrix
    : OpenKalman::internal::MatrixBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(has_dynamic_dimensions<NestedMatrix> or square_matrix<NestedMatrix>);
    static_assert(triangle_type != TriangleType::none);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, NestedMatrix>;

    static constexpr auto uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;

    static constexpr auto dim = row_dimension_of_v<NestedMatrix>;

  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    TriangularMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    TriangularMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    TriangularMatrix(const TriangularMatrix& other) : Base {other} {}


    /// Move constructor.
    TriangularMatrix(TriangularMatrix&& other) noexcept : Base {std::move(other)} {}


    /// Construct from a compatible triangular matrix object of the same TriangleType.
#ifdef __cpp_concepts
    template<typename  Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      ((eigen_triangular_expr<Arg> and triangle_type_of_v<Arg> == triangle_type) or
       (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_of_t<Arg>>) and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      ((eigen_triangular_expr<Arg> and triangle_type_of<Arg>::value == triangle_type) or
       (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
      (not eigen_diagonal_expr<NestedMatrix> or diagonal_matrix<nested_matrix_of_t<Arg>>) and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


      /// Construct from a compatible triangular matrix object if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
      template<typename Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
        ((eigen_triangular_expr<Arg> and triangle_type_of_v<Arg> == triangle_type) or
         (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
        eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_of_t<Arg>>) and
        requires(Arg&& arg) { NestedMatrix {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))}; }
#else
      template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
        ((eigen_triangular_expr<Arg> and triangle_type_of<Arg>::value == triangle_type) or
         (eigen_self_adjoint_expr<Arg> and diagonal_matrix<Arg>)) and
        eigen_diagonal_expr<NestedMatrix> and (not diagonal_matrix<nested_matrix_of_t<Arg>>) and
        std::is_constructible_v<NestedMatrix, decltype(to_diagonal(diagonal_of(nested_matrix(std::declval<Arg&&>()))))>,
        int> = 0>
#endif
      TriangularMatrix(Arg&& arg) noexcept : Base {to_diagonal(diagonal_of(nested_matrix(std::forward<Arg>(arg))))} {}


      /// Construct from an \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_diagonal_expr Arg> requires std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      eigen_diagonal_expr<Arg> and std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from an Eigen::TriangularView. \todo Factor in possibility of Eigen::ZeroDiag or Eigen::UnitDiag
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<Eigen3::eigen_TriangularView Arg> requires (triangle_type_of_v<Arg> == triangle_type) and
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<
      Eigen3::eigen_TriangularView<Arg> and (triangle_type_of<Arg>::value == triangle_type) and
      std::is_constructible_v<NestedMatrix, decltype(std::declval<Arg&&>().nestedExpression())>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a \ref triangular_matrix "triangular" \ref eigen_matrix.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires triangular_matrix<Arg> and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and triangular_matrix<Arg> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a non-triangular \ref eigen_matrix if NestedMatrix is \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not triangular_matrix<Arg>) and eigen_diagonal_expr<NestedMatrix> and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not triangular_matrix<Arg>) and
      eigen_diagonal_expr<NestedMatrix> and (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(
      (has_dynamic_dimensions<Arg> ? (assert(get_dimensions_of<0>(arg) == get_dimensions_of<1>(arg)), arg) : arg)))} {}


    /// Construct from a non-triangular \ref eigen_matrix if NestedMatrix is not \ref eigen_diagonal_expr.
#ifdef __cpp_concepts
    template<eigen_matrix Arg> requires (not triangular_matrix<Arg>) and (not eigen_diagonal_expr<NestedMatrix>) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and (not triangular_matrix<Arg>) and
      (not eigen_diagonal_expr<NestedMatrix>) and (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(
      (has_dynamic_dimensions<Arg> ? (assert(get_dimensions_of<0>(arg) == get_dimensions_of<1>(arg)), arg) : arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if triangle_type is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix,
          eigen_matrix_t<Scalar, OpenKalman::internal::constexpr_sqrt(sizeof...(Args)),
          OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>> or
        (diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix,
          eigen_matrix_t<Scalar, sizeof...(Args), 1>>)), int> = 0>
#endif
    TriangularMatrix(Args ... args)
      : Base {MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if NestedMatrix is not \ref eigen_diagonal_expr but triangle_type is TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix, eigen_matrix_t<Scalar, sizeof...(Args), 1>> or
       std::is_constructible_v<NestedMatrix,
         eigen_matrix_t<Scalar, OpenKalman::internal::constexpr_sqrt(sizeof...(Args)),
         OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>>), int> = 0>
#endif
    TriangularMatrix(Args ... args)
      : Base {MatrixTraits<typename MatrixTraits<NestedMatrix>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)} {}


    /// Copy assignment operator
    auto& operator=(const TriangularMatrix& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        if (this != &other)
        {
          this->nested_matrix().template triangularView<uplo>() = other.nested_matrix();
        }
      return *this;
    }


    /// Move assignment operator
    auto& operator=(TriangularMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another triangular matrix (must be the same triangle)
#ifdef __cpp_concepts
    template<eigen_triangular_expr Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      (row_dimension_of_v<Arg> == dim) and (triangle_type_of_v<Arg> == triangle_type) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      (row_dimension_of<Arg>::value == dim) and (triangle_type_of<Arg>::value == triangle_type) and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>> and
      (not (eigen_diagonal_expr<NestedMatrix> and triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>),
      int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix> or identity_matrix<NestedMatrix>)
      {}
      else if constexpr (eigen_diagonal_expr<NestedMatrix>)
      {
        this->nested_matrix().nested_matrix() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr (triangle_type == TriangleType::diagonal)
      {
        this->nested_matrix().diagonal() = diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else if constexpr(std::is_lvalue_reference_v<Arg>)
      {
        this->nested_matrix().template triangularView<uplo>() = nested_matrix(arg);
      }
      else
      {
        this->nested_matrix() = nested_matrix(std::forward<Arg>(arg));
      }
      return *this;
    }


    /// Assign from a general \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires (not eigen_triangular_expr<Arg>) and
      (triangle_type_of_v<Arg> == triangle_type) and (row_dimension_of_v<Arg> == dim) and
      modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and (not eigen_triangular_expr<Arg>) and
      (triangle_type_of<Arg>::value == triangle_type) and (row_dimension_of<Arg>::value == dim) and
      modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix> or identity_matrix<NestedMatrix>)
      {}
      else if constexpr (zero_matrix<Arg>)
      {
        this->nested_matrix() = make_zero_matrix_like(nested_matrix(arg));
      }
      else if constexpr (identity_matrix<Arg>)
      {
        this->nested_matrix() = make_identity_matrix_like(nested_matrix(arg));
      }
      else if constexpr (Eigen3::eigen_TriangularView<Arg>)
      {
        // \todo Factor in possibility of Eigen::ZeroDiag or Eigen::UnitDiag
        if constexpr (std::is_rvalue_reference_v<Arg>)
        {
          this->nested_matrix() = std::move(arg.nestedExpression());
        }
        else
        {
          this->nested_matrix().template triangularView<uplo>() = arg.nestedExpression();
        }
      }
      else
      {
        this->nested_matrix() = std::forward<Arg>(arg);
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<row_dimension_of<Arg>::value == dim, int> = 0>
#endif
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      view() += arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<row_dimension_of<Arg>::value == dim, int> = 0>
#endif
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      view() -= arg.nested_matrix();
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      view() *= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      view() /= s;
      return *this;
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (row_dimension_of_v<Arg> == dim)
#else
    template<typename Arg, std::enable_if_t<row_dimension_of<Arg>::value == dim, int> = 0>
#endif
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      auto v {view()};
      v = v * make_dense_writable_matrix_from(arg);
      return *this;
    }


    auto view()
    {
      static_assert(not Eigen3::eigen_diagonal_expr<NestedMatrix>);
      return this->nested_matrix().template triangularView<uplo>();
    }


    const auto view() const
    {
      static_assert(not Eigen3::eigen_diagonal_expr<NestedMatrix>);
      return this->nested_matrix().template triangularView<uplo>();
    }


  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<eigen_diagonal_expr M>
#else
  template<typename M, std::enable_if_t<eigen_diagonal_expr<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::diagonal>;


#ifdef __cpp_concepts
  template<eigen_matrix M>
#else
  template<typename M, std::enable_if_t<eigen_matrix<M>, int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<Eigen3::eigen_TriangularView M>
#else
  template<typename M, std::enable_if_t<Eigen3::eigen_TriangularView<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<nested_matrix_of_t<M>, triangle_type_of_v<M>>;


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M> requires diagonal_matrix<M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M> and diagonal_matrix<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<nested_matrix_of_t<M>>, TriangleType::diagonal>;


  /// If the arguments are a sequence of scalars, deduce a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<scalar_type Arg, scalar_type ... Args> requires (std::common_with<Arg, Args> and ...)
#else
    template<typename Arg, typename ... Args, std::enable_if_t<
    (scalar_type<Arg> and ... and scalar_type<Args>), int> = 0>
#endif
  TriangularMatrix(const Arg&, const Args& ...) -> TriangularMatrix<
    Eigen3::eigen_matrix_t<
      std::common_type_t<Arg, Args...>,
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args)),
      OpenKalman::internal::constexpr_sqrt(1 + sizeof...(Args))>,
    TriangleType::lower>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M> requires eigen_matrix<M> or eigen_diagonal_expr<M>
#else
  template<
    TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<eigen_matrix<M> or eigen_diagonal_expr<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<TriangleType t, eigen_triangular_expr M> requires (t == triangle_type_of_v<M>)
#else
  template<TriangleType t, typename M, std::enable_if_t<
    eigen_triangular_expr<M> and (t == triangle_type_of<M>::value), int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<t>(nested_matrix(std::forward<M>(m)));
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr M>
#else
  template<typename M, std::enable_if_t<eigen_triangular_expr<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return make_EigenTriangularMatrix<triangle_type_of_v<M>>(nested_matrix(std::forward<M>(m)));
  }

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

