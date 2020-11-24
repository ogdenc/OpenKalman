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
  template<coefficients Coeffs, typed_matrix_nestable NestedMatrix> requires
    (Coeffs::dimension == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coeffs, typename NestedMatrix>
#endif
  struct EuclideanMean : internal::TypedMatrixBase<EuclideanMean<Coeffs, NestedMatrix>,
    Coeffs, Axes<MatrixTraits<NestedMatrix>::columns>, NestedMatrix>
  {
    using Coefficients = Coeffs;
    using Base = internal::TypedMatrixBase<EuclideanMean, Coefficients, Axes<MatrixTraits<NestedMatrix>::columns>, NestedMatrix>;
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(Base::dimension == Coefficients::dimension);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);

    using Base::Base;

    /// Copy constructor.
    EuclideanMean(const EuclideanMean& other) : Base(other.nested_matrix()) {}

    /// Move constructor.
    EuclideanMean(EuclideanMean&& other) noexcept : Base(std::move(other).nested_matrix()) {}

    /// Construct from a compatible Euclidean-transformed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and euclidean_transformed<Arg>, int> = 0>
#endif
    EuclideanMean(Arg&& other) noexcept : Base(std::forward<Arg>(other).nested_matrix())
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
    EuclideanMean(Arg&& other) noexcept : Base(OpenKalman::to_Euclidean<Coefficients>(std::forward<Arg>(other).nested_matrix()))
    {
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
    }

    /// Construct from compatible typed matrix object.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) noexcept : Base(std::forward<Arg>(arg))
    {
      static_assert(MatrixTraits<Arg>::dimension == Base::dimension);
      static_assert(MatrixTraits<Arg>::columns == Base::columns);
    }

    /// Copy assignment operator.
    auto& operator=(const EuclideanMean& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
        this->nested_matrix() = other.nested_matrix();
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(EuclideanMean&& other)
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
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, typename Base::ColumnCoefficients>);
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(euclidean_transformed<Arg> or Coefficients::axes_only)
      {
        this->nested_matrix() = std::forward<Arg>(other).nested_matrix();
      }
      else
      {
        this->nested_matrix() = to_Euclidean<Coefficients>(std::forward<Arg>(other).nested_matrix());
      }
      return *this;
    }

    /// Increment from another EuclideanMean.
    auto& operator+=(const EuclideanMean& other)
    {
      this->nested_matrix() += other.nested_matrix();
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
      this->nested_matrix() += std::forward<Arg>(other).nested_matrix();
      return *this;
    }

    /// Add a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
    auto& operator+=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      static_assert(Coefficients::axes_only);
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){ col += arg().nested_matrix(); });
      return *this;
    }

    /// Decrement from another EuclideanMean.
    auto& operator-=(const EuclideanMean& other)
    {
      this->nested_matrix() -= other.nested_matrix();
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
      this->nested_matrix() -= std::forward<Arg>(other).nested_matrix();
      return *this;
    }

    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
    template<typename Arg, std::enable_if_t<distribution<Arg>, int> = 0>
    auto& operator-=(const Arg& arg) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      static_assert(Coefficients::axes_only);
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){ col -= arg().nested_matrix(); });
      return *this;
    }

  protected:
    template<typename C = Coefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept { return EuclideanMean<C, self_contained_t<Arg>>(std::forward<Arg>(arg)); }

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
  EuclideanMean(V&&) -> EuclideanMean<Axes<MatrixTraits<V>::dimension>, passable_t<V>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and euclidean_transformed<V>, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients, nested_matrix_t<V>>;


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
    decltype(to_Euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).nested_matrix()))>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /// Make a EuclideanMean object from a regular matrix object.
#ifdef __cpp_concepts
  template<typename Coefficients = void, typed_matrix_nestable V> requires std::same_as<Coefficients, void> or
    (coefficients<Coefficients> and MatrixTraits<V>::dimension == Coefficients::dimension)
#else
  template<typename Coefficients = void, typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
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
      return make_EuclideanMean<typename MatrixTraits<Arg>::RowCoefficients>(nested_matrix(std::forward<Arg>(arg)));
    else
      return make_EuclideanMean<typename MatrixTraits<Arg>::RowCoefficients>(nested_matrix(to_Euclidean<C>(std::forward<Arg>(arg))));
  }


  /// Make a default, self-contained EuclideanMean.
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable V> requires
    (coefficients<Coefficients> and Coefficients::dimension == MatrixTraits<V>::dimension)
#else
  template<typename Coefficients, typename V, std::enable_if_t<typed_matrix_nestable<V> and
    (Coefficients::dimension == MatrixTraits<V>::dimension), int> = 0>
#endif
  auto make_EuclideanMean()
  {
    constexpr auto rows = MatrixTraits<V>::dimension;
    return EuclideanMean<Coefficients, native_matrix_t<V, rows>>();
  }


  /// Make a default, self-contained EuclideanMean with axis coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
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
    using NestedMatrix = NestedType;
    static constexpr std::size_t dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr std::size_t columns = MatrixTraits<NestedMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static_assert(RowCoefficients::dimension == dimension);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = EuclideanMean<RowCoefficients, self_contained_t<NestedMatrix>>;

    /// Make from a regular matrix. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
      coefficients<CC> or std::same_as<CC, void>
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg,
      std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      static_assert(MatrixTraits<Arg>::dimension == RC::dimension);
      if constexpr(not std::is_void_v<CC>) static_assert(equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>);
      return EuclideanMean<RC, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity()
    {
      auto b = MatrixTraits<NestedMatrix>::identity();
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


} // namespace OpenKalman::internal


#endif //OPENKALMAN_EUCLIDEANMEAN_HPP
