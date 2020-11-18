/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EUCLIDEANMEAN_HPP
#define OPENKALMAN_EUCLIDEANMEAN_HPP

namespace OpenKalman
{
  //////////////////////////////////
  //        EuclideanMean         //
  //////////////////////////////////

#ifdef __cpp_concepts
  template<coefficients Coeffs, typed_matrix_base BaseMatrix> requires
    (Coeffs::dimension == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coeffs, typename BaseMatrix>
#endif
  struct EuclideanMean : internal::TypedMatrixBase<EuclideanMean<Coeffs, BaseMatrix>,
    Coeffs, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>
  {
    using Coefficients = Coeffs;
    using Base = internal::TypedMatrixBase<EuclideanMean, Coefficients, Axes<MatrixTraits<BaseMatrix>::columns>, BaseMatrix>;
    static_assert(typed_matrix_base<BaseMatrix>);
    static_assert(Base::dimension == Coefficients::dimension);

    using Base::Base;

    /// Copy constructor.
    EuclideanMean(const EuclideanMean& other) : Base(other.base_matrix()) {}

    /// Move constructor.
    EuclideanMean(EuclideanMean&& other) noexcept : Base(std::move(other).base_matrix()) {}

    /// Construct from a compatible Euclidean-transformed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg>, int> = 0>
#endif
    EuclideanMean(Arg&& other) noexcept : Base(std::forward<Arg>(other).base_matrix())
    {
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from a compatible non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not euclidean_transformed<Arg>, int> = 0>
#endif
    EuclideanMean(Arg&& other) noexcept : Base(OpenKalman::to_Euclidean<Coefficients>(std::forward<Arg>(other).base_matrix()))
    {
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from compatible typed matrix object.
#ifdef __cpp_concepts
    template<typed_matrix_base Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_base<Arg>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Copy assignment operator.
    auto& operator=(const EuclideanMean& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
        this->base_matrix() = other.base_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(EuclideanMean&& other)
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
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(euclidean_transformed<Arg> or Coefficients::axes_only)
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
#ifdef __cpp_concepts
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      static_assert(Coefficients::axes_only or euclidean_transformed<Arg>);
      this->base_matrix() += std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Add a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
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
#ifdef __cpp_concepts
    template<typed_matrix Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(MatrixTraits<Arg>::ColumnCoefficients::axes_only);
      static_assert(Coefficients::axes_only or euclidean_transformed<Arg>);
      this->base_matrix() -= std::forward<Arg>(other).base_matrix();
      return *this;
    }

    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      static_assert(Coefficients::axes_only);
      apply_columnwise(this->base_matrix(), [&arg](auto& col){ col -= arg().base_matrix(); });
      return *this;
    }

  protected:
    template<typename C = Coefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return EuclideanMean<C, self_contained_t<Arg>>(std::forward<Arg>(arg)); }

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
  EuclideanMean(V&&) -> EuclideanMean<Axes<MatrixTraits<V>::dimension>, passable_t<V>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and euclidean_transformed<V>, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients, typename MatrixTraits<V>::BaseMatrix>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \TODO Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>) and
    MatrixTraits<V>::ColumnCoefficients::axes_only and
    (MatrixTraits<V>::RowCoefficients::dimension == MatrixTraits<V>::dimension)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V> and
    MatrixTraits<V>::ColumnCoefficients::axes_only and
    MatrixTraits<V>::RowCoefficients::dimension == MatrixTraits<V>::dimension, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients,
    decltype(to_Euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).base_matrix()))>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /// Make a EuclideanMean object from a regular matrix object.
#ifdef __cpp_concepts
  template<typename Coefficients = void, typed_matrix_base V> requires std::same_as<Coefficients, void> or
    (coefficients<Coefficients> and MatrixTraits<V>::dimension == Coefficients::dimension)
#else
  template<typename Coefficients = void, typename V, std::enable_if_t<typed_matrix_base<V>, int> = 0>
#endif
  auto make_EuclideanMean(V&& arg) noexcept
  {
    constexpr auto rows = MatrixTraits<V>::dimension;
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>, Axes<rows>, Coefficients>;
    static_assert(Coeffs::dimension == rows);
    return EuclideanMean<Coeffs, passable_t<V>>(std::forward<V>(arg));
  }


  /// Make a EuclideanMean object from another typed matrix.
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto make_EuclideanMean(Arg&& arg) noexcept
  {
    static_assert(column_vector<Arg>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_EuclideanMean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(std::forward<Arg>(arg)));
    else
      return make_EuclideanMean<typename MatrixTraits<Arg>::RowCoefficients>(base_matrix(to_Euclidean<C>(std::forward<Arg>(arg))));
  }


  /// Make a default, self-contained EuclideanMean.
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_base V> requires
    (coefficients<Coefficients> and Coefficients::dimension == MatrixTraits<V>::dimension)
#else
  template<typename Coefficients, typename V, std::enable_if_t<typed_matrix_base<V> and
    (Coefficients::dimension == MatrixTraits<V>::dimension), int> = 0>
#endif
  auto make_EuclideanMean()
  {
    constexpr auto rows = MatrixTraits<V>::dimension;
    return EuclideanMean<Coefficients, native_matrix_t<V, rows>>();
  }


  /// Make a default, self-contained EuclideanMean with axis coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_base V>
#else
  template<typename V, std::enable_if_t<typed_matrix_base<V>, int> = 0>
#endif
  auto make_EuclideanMean()
  {
    return make_EuclideanMean<Axes<MatrixTraits<V>::dimension>, V>();
  }


  // --------------------- //
  //        Traits         //
  // --------------------- //

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
    using NativeMatrix = typename MatrixTraits<BaseMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = EuclideanMean<RowCoefficients, self_contained_t<BaseMatrix>>;

    /// Make from a regular matrix. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_base Arg> requires
      coefficients<CC> or std::same_as<CC, void>
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg,
      std::enable_if_t<typed_matrix_base<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::dimension);
      if constexpr(not std::is_void_v<CC>) static_assert(equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>);
      return EuclideanMean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity()
    {
      auto b = MatrixTraits<BaseMatrix>::identity();
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_EUCLIDEANMEAN_HPP
