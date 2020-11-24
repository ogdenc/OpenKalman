/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
  template<coefficients Coeffs, typed_matrix_base BaseMatrix> requires
    (Coeffs::size == MatrixTraits<BaseMatrix>::dimension) and (not std::is_rvalue_reference_v<BaseMatrix>)
#else
  template<typename Coeffs, typename BaseMatrix>
#endif
  struct Mean
    : internal::TypedMatrixBase<Mean<Coeffs, BaseMatrix>, Coeffs, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>
  {
    using Coefficients = Coeffs;
    using Base = internal::TypedMatrixBase<Mean, Coefficients, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>;
    static_assert(typed_matrix_base<BaseMatrix>);
    static_assert(Base::dimension == Coefficients::size);
    static_assert(not std::is_rvalue_reference_v<BaseMatrix>);
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this variable.

    /// Default constructor.
    Mean() : Base() {}

    /// Copy constructor.
    Mean(const Mean& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    Mean(Mean&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Construct from a compatible mean.
#ifdef __cpp_concepts
    template<mean Arg>
#else
    template<typename Arg, std::enable_if_t<mean<Arg>, int> = 0>
#endif
    Mean(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
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
    Mean(Arg&& other) noexcept : Mean(std::forward<Arg>(other).base_matrix())
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
    Mean(Arg&& other) noexcept : Base(from_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix()))
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }


    /// Construct from a typed matrix base.
#ifdef __cpp_concepts
    template<typed_matrix_base Arg> requires
      (not (internal::contains_nested_lvalue_reference<Arg> or
      not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>))
#else
    template<typename Arg, std::enable_if_t<typed_matrix_base<Arg> and not
      (internal::contains_nested_lvalue_reference<Arg> or
      not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>), int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base(wrap_angles<Coefficients>(std::forward<Arg>(arg)))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }


    /// Construct from a typed matrix base. For situations when angle wrapping should not occur.
#ifdef __cpp_concepts
    template<typed_matrix_base Arg> requires
      internal::contains_nested_lvalue_reference<Arg> or
      (not std::is_constructible_v<Base, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix_base<Arg> and
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
    Mean(Args ... args) : Mean(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator.
    auto& operator=(const Mean& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
        this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(Mean&& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
        this->base_matrix() = std::move(other).base_matrix();
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
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(mean<Arg>)
      {
        this->base_matrix() = std::forward<Arg>(other).base_matrix();
      }
      else if constexpr(euclidean_transformed<Arg>)
      {
        this->base_matrix() = OpenKalman::from_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix());
      }
      else
      {
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(std::forward<Arg>(other).base_matrix());
      }
      return *this;
    }

    /// Increment from another mean.
    auto& operator+=(const Mean& other)
    {
      if constexpr(Coefficients::axes_only)
        this->base_matrix() += other.base_matrix();
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() + other.base_matrix());
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
        this->base_matrix() += std::forward<Arg>(other).base_matrix();
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() + std::forward<Arg>(other).base_matrix());
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
      apply_columnwise(this->base_matrix(), [&arg](auto& col){
        if constexpr(Coefficients::axes_only)
          col += arg().base_matrix();
        else
          col = OpenKalman::wrap_angles<Coefficients>(col + arg().base_matrix());
      });
      return *this;
    }

    /// Decrement from another mean and wrap result.
    auto& operator-=(const Mean& other)
    {
      if constexpr(Coefficients::axes_only)
        this->base_matrix() -= other.base_matrix();
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() - other.base_matrix());
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
        this->base_matrix() -= std::forward<Arg>(other).base_matrix();
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() - std::forward<Arg>(other).base_matrix());
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
      apply_columnwise(this->base_matrix(), [&arg](auto& col){
        if constexpr(Coefficients::axes_only)
          col -= arg().base_matrix();
        else
          col = OpenKalman::wrap_angles<Coefficients>(col - arg().base_matrix());
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
        this->base_matrix() *= s;
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() * s);
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
        this->base_matrix() /= s;
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() / s);
      return *this;
    }

  protected:
    template<typename C = Coefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return Mean<C, self_contained_t<Arg>>(std::forward<Arg>(arg)); }

  public:
    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }
  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  /// Deduce template parameters from a typed matrix base, assuming axis-only coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_base V>
#else
  template<typename V, std::enable_if_t<typed_matrix_base<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<Axes<MatrixTraits<V>::dimension>, passable_t<V>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \TODO Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients, std::decay_t<decltype(
    wrap_angles<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \TODO Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V> requires MatrixTraits<V>::ColumnCoefficients::axes_only
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V> and
    MatrixTraits<V>::ColumnCoefficients::axes_only, int> = 0>
#endif
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients, std::decay_t<decltype(
    from_Euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /// Make a Mean object from a typed matrix base.
#ifdef __cpp_concepts
  template<typename Coefficients = void, typed_matrix_base Arg> requires std::same_as<Coefficients, void> or
    (coefficients<Coefficients> and MatrixTraits<Arg>::dimension == Coefficients::size)
#else
  template<typename Coefficients = void, typename Arg, std::enable_if_t<typed_matrix_base<Arg>, int> = 0>
#endif
  inline auto make_Mean(Arg&& arg) noexcept
  {
    constexpr auto rows = MatrixTraits<Arg>::dimension;
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>, Axes<rows>, Coefficients>;
    static_assert(Coeffs::size == rows);
    decltype(auto) b = wrap_angles<Coeffs>(std::forward<Arg>(arg));
    return Mean<Coeffs, passable_t<decltype(b)>>(b);
  }


  /// Make a Mean object from another typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto make_Mean(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_Mean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(from_Euclidean<C>(std::forward<Arg>(arg))));
    else
      return make_Mean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default, self-contained Mean
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_base V> requires
    (coefficients<Coefficients> and MatrixTraits<V>::dimension == Coefficients::size)
#else
  template<typename Coefficients, typename V, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_base<V> and
    (coefficients<Coefficients> and MatrixTraits<V>::dimension == Coefficients::size), int> = 0>
#endif
  inline auto make_Mean()
  {
    return Mean<Coefficients, native_matrix_t<V>>();
  }


  /// Make a default, self-contained Mean, with axis coefficients
#ifdef __cpp_concepts
  template<typed_matrix_base V>
#else
  template<typename V, std::enable_if_t<typed_matrix_base<V>, int> = 0>
#endif
  inline auto make_Mean()
  {
    return make_Mean<Axes<MatrixTraits<V>::dimension>, V>();
  }


  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename Coeffs, typename NestedType>
  struct MatrixTraits<Mean<Coeffs, NestedType>>
  {
    using BaseMatrix = NestedType;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static_assert(RowCoefficients::size == dimension);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<BaseMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Mean<RowCoefficients, self_contained_t<BaseMatrix>>;

    /// Make from a typed matrix base. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_base Arg> requires
      coefficients<CC> or std::same_as<CC, void>
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      typed_matrix_base<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::size);
      if constexpr(not std::is_void_v<CC>) static_assert(equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>);
      decltype(auto) b = wrap_angles<RC>(std::forward<Arg>(arg));
      return Mean<RC, std::decay_t<decltype(b)>>(b);
    }

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity()
    {
      auto b = MatrixTraits<BaseMatrix>::identity();
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_wrap_angles(MEAN_H
