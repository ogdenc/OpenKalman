/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to Mean.
 */
#ifndef OPENKALMAN_MEAN_HPP
#define OPENKALMAN_MEAN_HPP

namespace OpenKalman
{
  /////////////////////////
  //        Mean         //
  /////////////////////////

  /// A typed vector.
#ifdef __cpp_concepts
  template<coefficients Coeffs, typed_matrix_nestable NestedMatrix> requires
    (Coeffs::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coeffs, typename NestedMatrix>
#endif
  struct Mean
    : internal::TypedMatrixBase<Mean<Coeffs, NestedMatrix>, Coeffs, Axes<MatrixTraits<NestedMatrix>::columns>, NestedMatrix>
  {
    using Coefficients = Coeffs;
    using Base = internal::TypedMatrixBase<Mean, Coefficients, Axes<MatrixTraits<NestedMatrix>::columns>, NestedMatrix>;
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(Base::dimension == Coefficients::size);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this variable.

    /// Default constructor.
    Mean() : Base() {}

    /// Copy constructor.
    Mean(const Mean& other) : Base(other.nested_matrix()) {}

    /// Move constructor.
    Mean(Mean&& other) noexcept : Base(std::move(other).nested_matrix()) {}

    /// Construct from a compatible mean.
#ifdef __cpp_concepts
    template<mean Arg>
#else
    template<typename Arg, std::enable_if_t<mean<Arg>, int> = 0>
#endif
    Mean(Arg&& other) noexcept : Base(std::forward<Arg>(other).nested_matrix())
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from a compatible typed matrix or Euclidean-transformed mean.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and (not mean<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not euclidean_transformed<Arg> and
      not mean<Arg>, int> = 0>
#endif
    Mean(Arg&& other) noexcept : Mean(std::forward<Arg>(other).nested_matrix())
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from a compatible Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg>, int> = 0>
#endif
    Mean(Arg&& other) noexcept : Base(from_euclidean<Coefficients>(std::forward<Arg>(other).nested_matrix()))
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires
      (not (internal::contains_nested_lvalue_reference<Arg> or
      not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>))
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and not
      (internal::contains_nested_lvalue_reference<Arg> or
      not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>), int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base(wrap_angles<Coefficients>(std::forward<Arg>(arg)))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }


    /// Construct from a typed_matrix_nestable. For situations when angle wrapping should not occur.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires
      internal::contains_nested_lvalue_reference<Arg> or
      (not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (internal::contains_nested_lvalue_reference<Arg> or
      not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>), int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args>
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
#endif
    Mean(Args ... args) : Mean(MatrixTraits<NestedMatrix>::make(args...)) {}

    /// Copy assignment operator.
    auto& operator=(const Mean& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        this->nested_matrix() = other.nested_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(Mean&& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        this->nested_matrix() = std::move(other).nested_matrix();
      return *this;
    }

    /// Assign from a compatible typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(mean<Arg>)
      {
        this->nested_matrix() = std::forward<Arg>(other).nested_matrix();
      }
      else if constexpr(euclidean_transformed<Arg>)
      {
        this->nested_matrix() = OpenKalman::from_euclidean<Coefficients>(std::forward<Arg>(other).nested_matrix());
      }
      else
      {
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(std::forward<Arg>(other).nested_matrix());
      }
      return *this;
    }

    /// Increment from another mean.
    auto& operator+=(const Mean& other)
    {
      if constexpr(Coefficients::axes_only)
        this->nested_matrix() += other.nested_matrix();
      else
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(this->nested_matrix() + other.nested_matrix());
      return *this;
    }

    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      static_assert(not euclidean_transformed<Arg>);
      if constexpr(Coefficients::axes_only)
        this->nested_matrix() += std::forward<Arg>(other).nested_matrix();
      else
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(this->nested_matrix() + std::forward<Arg>(other).nested_matrix());
      return *this;
    }

    /// Add a stochastic value to each column of the mean, based on a distribution.
#ifdef __cpp_concepts
    template<typename Arg> requires distribution<Arg>
#else
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){
        if constexpr(Coefficients::axes_only)
          col += arg().nested_matrix();
        else
          col = OpenKalman::wrap_angles<Coefficients>(col + arg().nested_matrix());
      });
      return *this;
    }

    /// Decrement from another mean and wrap result.
    auto& operator-=(const Mean& other)
    {
      if constexpr(Coefficients::axes_only)
        this->nested_matrix() -= other.nested_matrix();
      else
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(this->nested_matrix() - other.nested_matrix());
      return *this;
    }

    /// Decrement from another typed matrix and wrap result.
#ifdef __cpp_concepts
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      static_assert(not euclidean_transformed<Arg>);
      if constexpr(Coefficients::axes_only)
        this->nested_matrix() -= std::forward<Arg>(other).nested_matrix();
      else
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(this->nested_matrix() - std::forward<Arg>(other).nested_matrix());
      return *this;
    }

    /// Subtract a stochastic value to each column of the mean, based on a distribution.
#ifdef __cpp_concepts
    template<typename Arg> requires distribution<Arg>
#else
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){
        if constexpr(Coefficients::axes_only)
          col -= arg().nested_matrix();
        else
          col = OpenKalman::wrap_angles<Coefficients>(col - arg().nested_matrix());
      });
      return *this;
    }

    /// Multiply by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(Coefficients::axes_only)
        this->nested_matrix() *= s;
      else
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(this->nested_matrix() * s);
      return *this;
    }

    /// Divide by a scale factor.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      if constexpr(Coefficients::axes_only)
        this->nested_matrix() /= s;
      else
        this->nested_matrix() = OpenKalman::wrap_angles<Coefficients>(this->nested_matrix() / s);
      return *this;
    }

  protected:
    template<typename C = Coefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return Mean<C, self_contained_t<Arg>>(std::forward<Arg>(arg)); }

  public:
    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }
  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  /// Deduce template parameters from a typed_matrix_nestable, assuming axis-only coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<Axes<MatrixTraits<V>::dimension>, passable_t<V>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \todo Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients, std::decay_t<decltype(
    wrap_angles<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).nested_matrix()))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V> requires MatrixTraits<V>::ColumnCoefficients::axes_only
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V> and
    MatrixTraits<V>::ColumnCoefficients::axes_only, int> = 0>
#endif
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients, std::decay_t<decltype(
    from_euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).nested_matrix()))>>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /**
   * \brief Make a Mean from a typed_matrix_nestable, specifying the row coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable M> requires
    (Coefficients::size == MatrixTraits<M>::dimension)
#else
  template<typename Coefficients, typename M, std::enable_if_t<coefficients<Coefficients> and
    typed_matrix_nestable<M> and (Coefficients::size == MatrixTraits<M>::dimension), int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    constexpr auto rows = MatrixTraits<M>::dimension;
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>, Axes<rows>, Coefficients>;
    decltype(auto) b = wrap_angles<Coeffs>(std::forward<M>(m));
    return Mean<Coeffs, passable_t<decltype(b)>>(b);
  }


  /**
   * \overload
   * \brief Make a Mean from a typed_matrix_nestable object, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    using Coeffs = Axes<MatrixTraits<M>::dimension>;
    return make_mean<Coeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Mean from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg>, int> = 0>
#endif
  inline auto make_mean(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_mean<C>(nested_matrix(from_euclidean<C>(std::forward<Arg>(arg))));
    else
      return make_mean<C>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a default, self-contained Mean.
   * \tparam Coefficients The coefficient types corresponding to the rows.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable M> requires
    (MatrixTraits<M>::dimension == Coefficients::size)
#else
  template<typename Coefficients, typename M, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::dimension == Coefficients::size), int> = 0>
#endif
  inline auto make_mean()
  {
    return Mean<Coefficients, native_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained Mean with default Axis coefficients.
   * \tparam M a typed_matrix_nestable on which the new mean is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_mean()
  {
    return make_mean<Axes<MatrixTraits<M>::dimension>, M>();
  }


  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename Coeffs, typename NestedType>
  struct MatrixTraits<Mean<Coeffs, NestedType>>
  {
    using NestedMatrix = NestedType;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static_assert(RowCoefficients::size == dimension);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Mean<RowCoefficients, self_contained_t<NestedMatrix>>;

    /// Make from a typed_matrix_nestable. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
      coefficients<CC> or std::same_as<CC, void>
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      typed_matrix_nestable<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::size);
      if constexpr(not std::is_void_v<CC>) static_assert(equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>);
      decltype(auto) b = wrap_angles<RC>(std::forward<Arg>(arg));
      return Mean<RC, std::decay_t<decltype(b)>>(b);
    }

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity()
    {
      auto b = MatrixTraits<NestedMatrix>::identity();
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_wrap_angles(MEAN_H
