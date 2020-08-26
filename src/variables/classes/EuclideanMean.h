/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EUCLIDEANMEAN_H
#define OPENKALMAN_EUCLIDEANMEAN_H

namespace OpenKalman
{
  //////////////////////////////////
  //        EuclideanMean         //
  //////////////////////////////////

  /// A typed vector.
  template<
    typename Coeffs,
    typename BaseMatrix>
  struct EuclideanMean : internal::TypedMatrixBase<EuclideanMean<Coeffs, BaseMatrix>,
    Coeffs, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>
  {
    using Coefficients = Coeffs;
    using Base = internal::TypedMatrixBase<EuclideanMean, Coefficients, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>;
    static_assert(is_typed_matrix_base_v<BaseMatrix>);
    static_assert(Base::dimension == Coefficients::dimension);

    using Base::Base;

    /// Copy constructor.
    EuclideanMean(const EuclideanMean& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    EuclideanMean(EuclideanMean&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Construct from a compatible Euclidean-transformed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and is_Euclidean_transformed_v<Arg>, int> = 0>
    EuclideanMean(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from a compatible non-Euclidean-transformed typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg> and not is_Euclidean_transformed_v<Arg>, int> = 0>
    EuclideanMean(Arg&& other) noexcept : Base(OpenKalman::to_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix()))
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from compatible typed matrix object.
    template<typename Arg, std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
    EuclideanMean(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Copy assignment operator.
    auto& operator=(const EuclideanMean& other)
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(EuclideanMean&& other)
    {
      if constexpr (not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>) if (this != &other)
        this->base_matrix() = std::move(other).base_matrix();
      return *this;
    }

    /// Assign from a compatible typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
      if constexpr (is_zero_v<BaseMatrix>)
      {
        static_assert(is_zero_v<Arg>);
      }
      else if constexpr (is_identity_v<BaseMatrix>)
      {
        static_assert(is_identity_v<Arg>);
      }
      else if constexpr(is_Euclidean_transformed_v<Arg> or Coefficients::axes_only)
      {
        this->base_matrix() = std::forward<Arg>(other).base_matrix();
      }
      else
      {
        this->base_matrix() = to_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix());
      }
      return *this;
    }

    /// Increment from another EuclideanMean.
    auto& operator+=(const EuclideanMean& other)
    {
      this->base_matrix() += other.base_matrix();
      return *this;
    }

    /// Increment from another typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      static_assert(Coefficients::axes_only or is_Euclidean_transformed_v<Arg>);
      this->base_matrix() += std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Add a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<is_distribution_v<Arg>, int> = 0>
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      static_assert(Coefficients::axes_only);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){ col += arg().base_matrix(); });
      return *this;
    }

    /// Decrement from another EuclideanMean.
    auto& operator-=(const EuclideanMean& other)
    {
      this->base_matrix() -= other.base_matrix();
      return *this;
    }

    /// Decrement from another typed matrix.
    template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(is_equivalent_v<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      static_assert(Coefficients::axes_only or is_Euclidean_transformed_v<Arg>);
      this->base_matrix() -= std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<is_distribution_v<Arg>, int> = 0>
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      static_assert(Coefficients::axes_only);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){ col -= arg().base_matrix(); });
      return *this;
    }

  protected:
    template<typename C = Coefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return EuclideanMean<C, strict_t<Arg>>(std::forward<Arg>(arg)); }

  public:
    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  /// Deduce template parameters from a typed matrix base, assuming axex-only coefficients.
  template<typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  EuclideanMean(V&&) -> EuclideanMean<Axes<MatrixTraits<V>::dimension>, lvalue_or_strict_t<V>>;

  /// Deduce template parameters from a Euclidean-transformed typed matrix.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V> and is_Euclidean_transformed_v<V>, int> = 0>
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<V>::BaseMatrix>;

  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
  template<typename V, std::enable_if_t<is_typed_matrix_v<V> and not is_Euclidean_transformed_v<V> and
    MatrixTraits<V>::ColumnCoefficients::axes_only, int> = 0>
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients,
    decltype(to_Euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /// Make a EuclideanMean object from a regular matrix object.
  template<typename Coefficients = void, typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  auto make_EuclideanMean(V&& arg) noexcept
  {
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>,
      Axes<MatrixTraits<V>::dimension>,
      Coefficients>;
    static_assert(MatrixTraits<V>::dimension == Coeffs::dimension);
    return EuclideanMean<Coeffs, lvalue_or_strict_t<V>>(std::forward<V>(arg));
  }


  /// Make a EuclideanMean object from another typed matrix.
  template<typename Arg, std::enable_if_t<is_typed_matrix_v<Arg>, int> = 0>
  inline auto make_EuclideanMean(Arg&& arg) noexcept
  {
    static_assert(is_column_vector_v<Arg>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(is_Euclidean_transformed_v<Arg>)
      return make_EuclideanMean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(std::forward<Arg>(arg)));
    else
      return make_EuclideanMean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(to_Euclidean<C>(std::forward<Arg>(arg))));
  }


  /// Make a default, strict EuclideanMean.
  template<typename Coefficients, typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  auto make_EuclideanMean()
  {
    static_assert(Coefficients::dimension == MatrixTraits<V>::dimension);
    constexpr auto rows = Coefficients::dimension;
    return EuclideanMean<Coefficients, typename MatrixTraits<V>::template StrictMatrix<rows>>();
  }


  /// Make a default, strict EuclideanMean with axis coefficients.
  template<typename V, std::enable_if_t<is_typed_matrix_base_v<V>, int> = 0>
  auto make_EuclideanMean()
  {
    return make_EuclideanMean<Axes<MatrixTraits<V>::dimension>, V>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename NestedType>
  struct MatrixTraits<EuclideanMean<Coeffs, NestedType>>
  {
    using BaseMatrix = NestedType;
    static constexpr std::size_t dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr std::size_t columns = MatrixTraits<BaseMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static_assert(RowCoefficients::dimension == dimension);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = typename MatrixTraits<BaseMatrix>::template StrictMatrix<rows, cols, S>;

    using Strict = EuclideanMean<RowCoefficients, typename MatrixTraits<BaseMatrix>::Strict>;

    /// Make from a regular matrix. If CC is specified, it must be axes-only.
    template<typename RC = RowCoefficients, typename CC = void, typename Arg,
      std::enable_if_t<is_typed_matrix_base_v<Arg>, int> = 0>
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::dimension);
      if constexpr(not std::is_void_v<CC>) static_assert(is_equivalent_v<CC, Axes<MatrixTraits<Arg>::columns>>);
      return EuclideanMean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity()
    {
      auto b = MatrixTraits<BaseMatrix>::identity();
      return TypedMatrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_EUCLIDEANMEAN_H
