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
 * \brief Definitions for TriangularMatrix
 */

#ifndef OPENKALMAN_TRIANGULARMATRIX_HPP
#define OPENKALMAN_TRIANGULARMATRIX_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> NestedMatrix, TriangleType triangle_type> requires (triangle_type != TriangleType::none)
#else
  template<typename NestedMatrix, TriangleType triangle_type>
#endif
  struct TriangularMatrix
    : OpenKalman::internal::MatrixBase<TriangularMatrix<NestedMatrix, triangle_type>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(square_matrix<NestedMatrix, Likelihood::maybe>);
    static_assert(triangle_type != TriangleType::none);
#endif

  private:

    using Base = OpenKalman::internal::MatrixBase<TriangularMatrix, NestedMatrix>;

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
    template<square_matrix<Likelihood::maybe> Arg> requires (not triangular_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      (not triangular_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
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
      std::is_constructible<NestedMatrix, Arg&&>::value, int> = 0>
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
      std::is_constructible<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>::value, int> = 0>
#endif
    TriangularMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a non-triangular square matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      (dimensions_match<Arg, 0>()) and (dimensions_match<Arg, 1>()) and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      (not triangular_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
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
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args...args) { NestedMatrix {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type != TriangleType::diagonal or diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    TriangularMatrix(Args...args)
      : Base {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a triangular matrix.
     * \note Operative if NestedMatrix is not \ref eigen_diagonal_expr but triangle_type is TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar>...Args> requires (sizeof...(Args) > 0) and
      (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        MatrixTraits<typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and (triangle_type == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    TriangularMatrix(Args...args)
      : Base {MatrixTraits<typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)} {}


    /// Assign from another \ref triangular_matrix.
#ifdef __cpp_concepts
    template<triangular_matrix Arg> requires (triangle_type_of_v<Arg> == triangle_type or diagonal_matrix<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, TriangularMatrix>) and
      maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not constant_diagonal_matrix<NestedMatrix> or
        requires { requires constant_diagonal_coefficient_v<NestedMatrix> == constant_diagonal_coefficient_v<Arg>; }) and
      (not (diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<triangular_matrix<Arg> and
      (triangle_type_of_v<Arg> == triangle_type or diagonal_matrix<Arg>) and
      (not std::is_base_of_v<TriangularMatrix, std::decay_t<Arg>>) and maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not constant_diagonal_matrix<NestedMatrix> or constant_diagonal_matrix<Arg>) and
      (not (diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal) or diagonal_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (not constant_diagonal_matrix<NestedMatrix>)
        set_triangle<triangle_type>(this->nested_matrix(), std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator+=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() + std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator-=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() - std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() * s);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      set_triangle<triangle_type>(this->nested_matrix(), this->nested_matrix() / s);
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg>
#else
    template<typename Arg, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator*=(const TriangularMatrix<Arg, triangle_type>& arg)
    {
      contract_in_place(this->nested_matrix(), make_dense_writable_matrix_from(arg));
      return *this;
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<triangular_matrix M>
#else
  template<typename M, std::enable_if_t<triangular_matrix<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<M>, triangle_type_of_v<M>>;


#ifdef __cpp_concepts
  template<hermitian_adapter M> requires (not triangular_matrix<M>)
#else
  template<typename M, std::enable_if_t<hermitian_adapter<M> and not triangular_matrix<M>, int> = 0>
#endif
  TriangularMatrix(M&&) -> TriangularMatrix<passable_t<nested_matrix_of_t<M>>, hermitian_adapter_type_of_v<M>>;


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

} // namespace OpenKalman


#endif //OPENKALMAN_TRIANGULARMATRIX_HPP

