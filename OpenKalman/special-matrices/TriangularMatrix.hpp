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

    static constexpr auto dim = dynamic_dimension<NestedMatrix, 0> ? index_dimension_of_v<NestedMatrix, 1> :
      index_dimension_of_v<NestedMatrix, 0>;

    template<typename Arg, std::size_t N>
    static bool constexpr dimensions_match()
    {
      return dynamic_dimension<Arg, N> or dim == dynamic_size or index_dimension_of_v<Arg, N> == dim;
    }

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


    /// Construct from a triangular matrix adapter if NestedMatrix is non-diagonal.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      (triangle_type_of_v<Arg> == triangle_type or diagonal_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      (dimensions_match<nested_matrix_of_t<Arg>, 0>()) and (dimensions_match<nested_matrix_of_t<Arg>, 1>()) and
# if OPENKALMAN_CPP_FEATURE_CONCEPTS
      requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } //-- not accepted in GCC 10.1.0
# else
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
# endif
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      (triangle_type_of<Arg>::value == triangle_type or diagonal_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      (dimensions_match<typename nested_matrix_of<Arg>::type, 0>()) and (dimensions_match<typename nested_matrix_of<Arg>::type, 1>()) and
      std::is_constructible<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular or square matrix if NestedMatrix is non-diagonal.
#ifdef __cpp_concepts
    template<typename Arg> requires (not triangular_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(not triangular_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>)
          {
            if (get_index_dimension_of<0>(arg) != get_index_dimension_of<1>(arg)) throw std::domain_error {
              "Argument to TriangularMatrix must be a square matrix, but the argument has dimensions " +
              std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
              " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          }
          return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))
    } {}


    /// Construct from a triangular, non-adapter matrix.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires
      (triangle_type_of_v<Arg> == triangle_type or diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg> or (diagonal_matrix<NestedMatrix> and diagonal_matrix<Arg>)) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and
      (triangle_type_of<Arg>::value == triangle_type or diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg> or (diagonal_matrix<NestedMatrix> and diagonal_matrix<Arg>)) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a diagonal matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      diagonal_matrix<NestedMatrix> and (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      (not std::constructible_from<NestedMatrix, Arg&&>) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and
      diagonal_matrix<NestedMatrix> and (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      (not std::is_constructible_v<NestedMatrix, Arg&&>) and
      std::is_constructible<NestedMatrix, decltype(to_diagonal(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular square matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<typename Arg> requires (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      (has_dynamic_dimensions<Arg> or square_matrix<Arg>) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      std::is_constructible<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    explicit TriangularMatrix(Arg&& arg) : Base {
      [](Arg&& arg) -> decltype(auto) {
          if constexpr (has_dynamic_dimensions<Arg>)
          {
            if (get_index_dimension_of<0>(arg) != get_index_dimension_of<1>(arg)) throw std::domain_error {
              "Argument to TriangularMatrix must be a square matrix, but the argument has dimensions " +
              std::to_string(get_index_dimension_of<0>(arg)) + "×" + std::to_string(get_index_dimension_of<1>(arg)) +
              " in " + __func__ + " at line " + std::to_string(__LINE__) + " of " + __FILE__};
          }
          return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))
    } {}


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
  template<typename M> requires eigen_zero_expr<M> or eigen_constant_expr<M>
#else
  template<typename M, std::enable_if_t<eigen_zero_expr<M> or eigen_constant_expr<M>, int> = 0>
#endif
  explicit TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, TriangleType::lower>;


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr M> requires diagonal_matrix<M>
#else
  template<typename M, std::enable_if_t<eigen_self_adjoint_expr<M> and diagonal_matrix<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<nested_matrix_of_t<M>>, TriangleType::diagonal>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M> requires (not triangular_matrix<M>)
#else
  template<
    TriangleType t = TriangleType::lower, typename M, std::enable_if_t<not triangular_matrix<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    return TriangularMatrix<passable_t<M>, t> (std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<TriangleType t, triangular_matrix M> requires (t == triangle_type_of_v<M> or diagonal_matrix<M>)
#else
  template<TriangleType t, typename M, std::enable_if_t<triangular_matrix<M> and
    (t == triangle_type_of<M>::value or diagonal_matrix<M>), int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    if constexpr (eigen_triangular_expr<M>)
      return make_EigenTriangularMatrix<t>(nested_matrix(std::forward<M>(m)));
    else
      return make_EigenTriangularMatrix<t>(std::forward<M>(m));
  }


#ifdef __cpp_concepts
  template<triangular_matrix M>
#else
  template<typename M, std::enable_if_t<triangular_matrix<M>, int> = 0>
#endif
  auto make_EigenTriangularMatrix(M&& m)
  {
    if constexpr (eigen_triangular_expr<M>)
      return make_EigenTriangularMatrix<triangle_type_of_v<M>>(nested_matrix(std::forward<M>(m)));
    else
      return make_EigenTriangularMatrix<triangle_type_of_v<M>>(std::forward<M>(m));
  }

} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_EIGEN3_TRIANGULARMATRIX_HPP

