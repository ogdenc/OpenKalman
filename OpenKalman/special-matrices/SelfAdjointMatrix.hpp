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
 * \brief SelfAdjointMatrix and related definitions.
 */

#ifndef OPENKALMAN_SELFADJOINTMATRIX_HPP
#define OPENKALMAN_SELFADJOINTMATRIX_HPP

namespace OpenKalman
{
  using namespace OpenKalman::internal;

#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> NestedMatrix, TriangleType storage_triangle> requires
    (not constant_matrix<NestedMatrix> or real_axis_number<constant_coefficient<NestedMatrix>>) and
    (not constant_diagonal_matrix<NestedMatrix> or real_axis_number<constant_diagonal_coefficient<NestedMatrix>>) and
    (storage_triangle != TriangleType::none)
#else
  template<typename NestedMatrix, TriangleType storage_triangle>
#endif
  struct SelfAdjointMatrix
    : OpenKalman::internal::MatrixBase<SelfAdjointMatrix<NestedMatrix, storage_triangle>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(square_matrix<NestedMatrix, Likelihood::maybe>);
    static_assert([]{if constexpr (constant_matrix<NestedMatrix>) return real_axis_number<constant_coefficient<NestedMatrix>>; else return true; }());
    static_assert([]{if constexpr (constant_diagonal_matrix<NestedMatrix>) return real_axis_number<constant_diagonal_coefficient<NestedMatrix>>; else return true; }());
    static_assert(storage_triangle != TriangleType::none);
#endif


  private:

    using Base = OpenKalman::internal::MatrixBase<SelfAdjointMatrix, NestedMatrix>;

    static constexpr auto dim =
      dynamic_rows<NestedMatrix> ? column_dimension_of_v<NestedMatrix> : row_dimension_of_v<NestedMatrix>;


  public:

    using Scalar = scalar_type_of_t<NestedMatrix>;


    /// Default constructor.
#ifdef __cpp_concepts
    SelfAdjointMatrix() requires std::default_initializable<NestedMatrix> and (not has_dynamic_dimensions<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<
      std::is_default_constructible_v<T> and (not has_dynamic_dimensions<NestedMatrix>), int> = 0>
    SelfAdjointMatrix()
#endif
      : Base {} {}


    /// Copy constructor.
    SelfAdjointMatrix(const SelfAdjointMatrix& other) : Base {other} {}


    /// Move constructor.
    SelfAdjointMatrix(SelfAdjointMatrix&& other) noexcept : Base {std::move(other)} {}


    /// Construct from a diagonal matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      diagonal_matrix<NestedMatrix> and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      diagonal_matrix<Arg> and diagonal_matrix<NestedMatrix> and
      std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {diagonal_of(std::forward<Arg>(arg))} {}


    /// Construct from a diagonal matrix if NestedMatrix is not diagonal.
#ifdef __cpp_concepts
    template<diagonal_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (not diagonal_matrix<NestedMatrix> or
        not requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      diagonal_matrix<Arg> and (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      (not diagonal_matrix<NestedMatrix> or
        not std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian, non-diagonal wrapper of the same storage type
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      (not diagonal_matrix<Arg>) and has_nested_matrix<Arg> and
      (hermitian_adapter_type_of<Arg>::value == storage_triangle) and square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
      //alt: requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } -- not accepted in GCC 10
#else
    template<typename Arg, std::enable_if_t<
      hermitian_matrix<Arg> and (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and
      (not diagonal_matrix<Arg>) and has_nested_matrix<Arg> and
      (hermitian_adapter_type_of<Arg>::value == storage_triangle) and square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a hermitian, non-diagonal wrapper of the opposite storage type
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      has_nested_matrix<Arg> and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      requires(Arg&& arg) { NestedMatrix {transpose(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<
      hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      has_nested_matrix<Arg> and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      square_matrix<nested_matrix_of_t<Arg>, Likelihood::maybe> and
      std::is_constructible_v<NestedMatrix, decltype(transpose(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a hermitian matrix of the same storage type and is not a wrapper
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value == storage_triangle) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value == storage_triangle) and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {std::forward<Arg>(arg)} {}


    /// Construct from a hermitian matrix, of the opposite storage type, that is not a wrapper
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      requires(Arg&& arg) { NestedMatrix {transpose(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and (not diagonal_matrix<Arg>) and
      (not has_nested_matrix<Arg>) and (hermitian_adapter_type_of<Arg>::value != storage_triangle) and
      std::is_constructible_v<NestedMatrix, decltype(transpose(std::declval<Arg&&>()))>, int> = 0>
#endif
    SelfAdjointMatrix(Arg&& arg) noexcept : Base {transpose(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedMatrix is not diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not hermitian_matrix<Arg>) and (not diagonal_matrix<NestedMatrix>) and
      std::constructible_from<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
      not hermitian_matrix<Arg> and not eigen_diagonal_expr<NestedMatrix> and
      std::is_constructible_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and dim != dynamic_size) assert(get_index_dimension_of<0>(arg) == dim);
        if constexpr (dynamic_columns<Arg> and dim != dynamic_size) assert(get_index_dimension_of<1>(arg) == dim);
        return std::forward<Arg>(arg);
      }(std::forward<Arg>(arg))} {}


    /// Construct from a non-hermitian matrix if NestedMatrix is diagonal.
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> Arg> requires (not hermitian_matrix<Arg>) and diagonal_matrix<NestedMatrix> and
      requires(Arg&& arg) { NestedMatrix {diagonal_of(std::forward<Arg>(arg))}; }
#else
    template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and (not hermitian_matrix<Arg>) and
      diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix, decltype(diagonal_of(std::declval<Arg&&>()))>, int> = 0>
#endif
    explicit SelfAdjointMatrix(Arg&& arg) noexcept
      : Base {[](Arg&& arg) -> decltype(auto) {
        if constexpr (dynamic_rows<Arg> and dim != dynamic_size) assert(get_index_dimension_of<0>(arg) == dim);
        if constexpr (dynamic_columns<Arg> and dim != dynamic_size) assert(get_index_dimension_of<1>(arg) == dim);
        return diagonal_of(std::forward<Arg>(arg));
      }(std::forward<Arg>(arg))} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \note Operative if storage_triangle is not TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
    requires (sizeof...(Args) > 0) and (storage_triangle != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (sizeof...(Args) > 0) and
      (storage_triangle != TriangleType::diagonal or diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix,
        untyped_dense_writable_matrix_t<NestedMatrix, Scalar, static_cast<std::size_t>(constexpr_sqrt(sizeof...(Args))), static_cast<std::size_t>(constexpr_sqrt(sizeof...(Args)))>> or
        (diagonal_matrix<NestedMatrix> and std::is_constructible_v<NestedMatrix,
          untyped_dense_writable_matrix_t<NestedMatrix, Scalar, sizeof...(Args), 1>>)), int> = 0>
#endif
    SelfAdjointMatrix(Args ... args)
      : Base {MatrixTraits<std::decay_t<NestedMatrix>>::make(static_cast<const Scalar>(args)...)} {}


    /**
     * \brief Construct from a list of scalar coefficients, in row-major order.
     * \details This assumes, without checking, that the coefficients represent a self-adjoint matrix.
     * \note Operative if NestedMatrix is not a \ref diagonal_matrix but storage_triangle is TriangleType::diagonal.
     * \tparam Args List of scalar values.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<std::convertible_to<const Scalar> ... Args>
    requires (sizeof...(Args) > 0) and
      (storage_triangle == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      requires(Args ... args) { NestedMatrix {
        MatrixTraits<typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>>::make(
          static_cast<const Scalar>(args)...)}; }
#else
    template<typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and
      std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      (storage_triangle == TriangleType::diagonal) and (not diagonal_matrix<NestedMatrix>) and
      (std::is_constructible_v<NestedMatrix, untyped_dense_writable_matrix_t<NestedMatrix, Scalar, sizeof...(Args), 1>> or
       std::is_constructible_v<NestedMatrix,
         untyped_dense_writable_matrix_t<NestedMatrix, Scalar, static_cast<std::size_t>(constexpr_sqrt(sizeof...(Args))), static_cast<std::size_t>(constexpr_sqrt(sizeof...(Args)))>>), int> = 0>
#endif
    SelfAdjointMatrix(Args ... args)
      : Base {MatrixTraits<typename MatrixTraits<std::decay_t<NestedMatrix>>::template DiagonalMatrixFrom<>>::make(
        static_cast<const Scalar>(args)...)} {}


    /** Copy assignment operator
     * \param other Another SelfAdjointMatrix
     * \return Reference to this.
     */
    auto& operator=(const SelfAdjointMatrix& other)
    {
      if constexpr (not constant_matrix<NestedMatrix> and not constant_diagonal_matrix<NestedMatrix>)
        if (this != &other)
        {
          set_triangle<storage_triangle>(this->nested_matrix(), other.nested_matrix());
        }
      return *this;
    }


    /** Move assignment operator
     * \param other A SelfAdjointMatrix temporary value.
     * \return Reference to this.
     */
    auto& operator=(SelfAdjointMatrix&& other) noexcept
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from another \ref hermitian_matrix.
#ifdef __cpp_concepts
    template<hermitian_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, SelfAdjointMatrix>) and
      maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not constant_matrix<NestedMatrix> or
        requires { requires constant_coefficient_v<NestedMatrix> == constant_coefficient_v<Arg>; }) and
      (not constant_diagonal_matrix<NestedMatrix> or
        requires { requires constant_diagonal_coefficient_v<NestedMatrix> == constant_diagonal_coefficient_v<Arg>; }) and
      (not (diagonal_matrix<NestedMatrix> or storage_triangle == TriangleType::diagonal) or diagonal_matrix<Arg>)
#else
    template<typename Arg, std::enable_if_t<hermitian_matrix<Arg> and
      (not std::is_base_of_v<SelfAdjointMatrix, std::decay_t<Arg>>) and maybe_has_same_shape_as<NestedMatrix, Arg> and
      (not constant_matrix<NestedMatrix> or constant_matrix<Arg>) and
      (not constant_diagonal_matrix<NestedMatrix> or constant_diagonal_matrix<Arg>) and
      (not (diagonal_matrix<NestedMatrix> or storage_triangle == TriangleType::diagonal) or diagonal_matrix<Arg>), int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
#ifndef __cpp_concepts
      if constexpr (constant_matrix<NestedMatrix>)
        static_assert(constant_coefficient_v<NestedMatrix> == constant_coefficient_v<Arg>);
      if constexpr (constant_diagonal_matrix<NestedMatrix>)
        static_assert(constant_diagonal_coefficient_v<NestedMatrix> == constant_diagonal_coefficient_v<Arg>);
#endif
      if constexpr (not constant_matrix<NestedMatrix> and not constant_diagonal_matrix<NestedMatrix>)
        set_triangle<storage_triangle>(this->nested_matrix(), std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator+=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() + std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<maybe_has_same_shape_as<NestedMatrix> Arg, TriangleType t>
#else
    template<typename Arg, TriangleType t, std::enable_if_t<maybe_has_same_shape_as<Arg, NestedMatrix>, int> = 0>
#endif
    auto& operator-=(const SelfAdjointMatrix<Arg, t>& arg)
    {
      set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() - std::forward<Arg>(arg));
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() * s);
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      set_triangle<storage_triangle>(this->nested_matrix(), this->nested_matrix() / s);
      return *this;
    }

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<hermitian_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<hermitian_matrix<Arg>, int> = 0>
#endif
  explicit SelfAdjointMatrix(Arg&&) -> SelfAdjointMatrix<passable_t<Arg>, hermitian_adapter_type_of_v<Arg>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

#ifdef __cpp_concepts
  template<TriangleType t = TriangleType::lower, typename M>
  requires (not hermitian_matrix<M>) and (t != TriangleType::none)
#else
  template<
    TriangleType t = TriangleType::lower, typename M,
    std::enable_if_t<(not hermitian_matrix<M>) and (t != TriangleType::none), int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }


#ifdef __cpp_concepts
  template<hermitian_matrix M>
#else
  template<typename M, std::enable_if_t<hermitian_matrix<M>, int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    constexpr TriangleType t = hermitian_adapter_type_of_v<M>;
    if constexpr(hermitian_adapter<M>)
      return make_EigenSelfAdjointMatrix<t>(nested_matrix(std::forward<M>(m)));
    else
      return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }


#ifdef __cpp_concepts
  template<TriangleType t, hermitian_matrix M> requires (t != TriangleType::none)
#else
  template<TriangleType t, typename M, std::enable_if_t<hermitian_matrix<M> and (t != TriangleType::none), int> = 0>
#endif
  auto
  make_EigenSelfAdjointMatrix(M&& m)
  {
    if constexpr(hermitian_adapter<M>)
    {
      if constexpr (t == hermitian_adapter_type_of_v<M> or diagonal_matrix<M>)
        return make_EigenSelfAdjointMatrix<t>(nested_matrix(std::forward<M>(m)));
      else
        return make_EigenSelfAdjointMatrix<t>(adjoint(nested_matrix(std::forward<M>(m))));
    }
    else return SelfAdjointMatrix<passable_t<M>, t> {std::forward<M>(m)};
  }

} // OpenKalman



#endif //OPENKALMAN_SELFADJOINTMATRIX_HPP

