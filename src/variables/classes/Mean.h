/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MEAN_H
#define OPENKALMAN_MEAN_H

#include "variables/support/TypedMatrixBase.h"

namespace OpenKalman
{
  /////////////////////////
  //        Mean         //
  /////////////////////////

  /// A typed vector.
  template<typename Coeffs, typename BaseMatrix>
  struct Mean : internal::TypedMatrixBase<Mean<Coeffs, BaseMatrix>,
    Coeffs, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>
  {
    using Coefficients = Coeffs;
    using Base = internal::TypedMatrixBase<Mean, Coefficients, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>;
    static_assert(is_typed_matrix_base_v<BaseMatrix>);
    static_assert(Base::dimension == Coefficients::size);
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this variable.
    using Base::base_matrix;

    /// Default constructor.
    Mean() : Base() {}

    /// Copy constructor.
    Mean(const Mean& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    Mean(Mean&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Construct from a typed matrix base.
    template<typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg> and
      std::is_constructible_v<BaseMatrix, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>, int> = 0>
    Mean(Arg&& arg) noexcept : Base(wrap_angles<Coefficients>(std::forward<Arg>(arg)))
    {
      static_assert(Coefficients::axes_only or not std::is_rvalue_reference_v<BaseMatrix>,
        "The base matrix of a Mean cannot be an rvalue reference if one of the coefficient types is not an Axis "
        "(because wrapping would be impossible).");
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Construct from a typed matrix base. For zero or identity matrices.
    template<typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg> and
      not std::is_constructible_v<BaseMatrix, decltype(wrap_angles<Coefficients>(std::declval<Arg>()))>, int> = 0>
    Mean(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Construct from a compatible typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and not is_Euclidean_transformed_v<Arg>, int> = 0>
    Mean(Arg&& other) noexcept : Mean(std::forward<Arg>(other).base_matrix())
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from a compatible Euclidean-transformed typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and is_Euclidean_transformed_v<Arg>, int> = 0>
    Mean(Arg&& other) noexcept : Base(from_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix()))
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from a list of coefficients.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    Mean(Args ... args) : Mean(MatrixTraits<BaseMatrix>::make(args...)) {}

    /// Copy assignment operator.
    Mean& operator=(const Mean& other)
    {
      this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    Mean& operator=(Mean&& other)
    {
      if (this != &other) this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
      if constexpr (std::is_same_v<std::decay_t<Arg>, Mean>) if (this == &other) return *this;
      if constexpr(is_mean_v<Arg>)
        this->base_matrix() = std::forward<Arg>(other).base_matrix();
      else if constexpr(is_Euclidean_transformed_v<Arg>)
        this->base_matrix() = OpenKalman::from_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix());
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(std::forward<Arg>(other).base_matrix());
      return *this;
    }

    /// Increment from another mean.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
      static_assert(not is_Euclidean_transformed_v<Arg>);
      if constexpr(Coefficients::axes_only)
        this->base_matrix() += std::forward<Arg>(other).base_matrix();
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() + std::forward<Arg>(other).base_matrix());
      return *this;
    }

    /// Add a stochastic value to each column of the mean, based on a distribution.
    template<typename Arg, std::enable_if_t<is_distribution_v<Arg>, int> = 0>
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){
        if constexpr(Coefficients::axes_only)
          col += arg().base_matrix();
        else
          col = OpenKalman::wrap_angles<Coefficients>(col + arg().base_matrix());
      });
      return *this;
    }

    /// Decrement from another mean.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
      static_assert(not is_Euclidean_transformed_v<Arg>);
      if constexpr(Coefficients::axes_only)
        this->base_matrix() -= std::forward<Arg>(other).base_matrix();
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() - std::forward<Arg>(other).base_matrix());
      return *this;
    }

    /// Subtract a stochastic value to each column of the mean, based on a distribution.
    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){
        if constexpr(Coefficients::axes_only)
          col -= arg().base_matrix();
        else
          col = OpenKalman::wrap_angles<Coefficients>(col - arg().base_matrix());
      });
      return *this;
    }

    /// Multiply by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
    auto& operator*=(const S s)
    {
      if constexpr(Coefficients::axes_only)
        this->base_matrix() *= s;
      else
        this->base_matrix() = OpenKalman::wrap_angles<Coefficients>(this->base_matrix() * s);
      return *this;
    }

    /// Divide by a scale factor.
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
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
    make(Arg&& arg) noexcept { return Mean<C, std::decay_t<Arg>>(std::forward<Arg>(arg)); }

  public:
    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  /// Deduce template parameters from a typed matrix base, assuming axis-only coefficients.
  template<typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  Mean(V&&) -> Mean<Axes<MatrixTraits<V>::dimension>, std::decay_t<V>>;

  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V> and not is_Euclidean_transformed_v<V>, int> = 0>
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients,
    decltype(wrap_angles<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>;

  /// Deduce template parameters from a Euclidean-transformed typed matrix.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V> and is_Euclidean_transformed_v<V> and
    MatrixTraits<V>::ColumnCoefficients::axes_only, int> = 0>
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients,
    decltype(from_Euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /// Make a Mean object from a typed matrix base.
  template<typename Coefficients = void, typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
  inline auto make_Mean(Arg&& arg) noexcept
  {
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>, Axes<MatrixTraits<Arg>::dimension>, Coefficients>;
    static_assert(MatrixTraits<Arg>::dimension == Coeffs::size);
    decltype(auto) b = wrap_angles<Coeffs>(std::forward<Arg>(arg));
    return Mean<Coeffs, std::decay_t<decltype(b)>>(b);
  }


  /// Make a Mean object from another typed matrix.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto make_Mean(Arg&& arg) noexcept
  {
    static_assert(is_column_vector_v<Arg>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(is_Euclidean_transformed_v<Arg>)
      return make_Mean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(from_Euclidean<C>(std::forward<Arg>(arg))));
    else
      return make_Mean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default, strict Mean
  template<typename Coefficients, typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  inline auto make_Mean()
  {
    static_assert(Coefficients::size == MatrixTraits<V>::dimension);
    return Mean<Coefficients, typename MatrixTraits<V>::template StrictMatrix<>>();
  }


  /// Make a default, strict Mean, with axis coefficients
  template<typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  inline auto make_Mean()
  {
    return make_Mean<Axes<MatrixTraits<V>::dimension>, V>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename NestedType>
  struct MatrixTraits<OpenKalman::Mean<Coeffs, NestedType>>
  {
    using BaseMatrix = NestedType;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = MatrixTraits<BaseMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static_assert(RowCoefficients::size == dimension);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    /// Make from a typed matrix base. If CC is specified, it must be axes-only.
    template<typename RC = RowCoefficients, typename CC = void, typename Arg,
      std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::size);
      if constexpr(not std::is_void_v<CC>) static_assert(is_equivalent_v<CC, Axes<MatrixTraits<Arg>::columns>>);
      decltype(auto) b = wrap_angles<RC>(std::forward<Arg>(arg));
      return Mean<RC, std::decay_t<decltype(b)>>(b);
    }

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_wrap_angles(MEAN_H
