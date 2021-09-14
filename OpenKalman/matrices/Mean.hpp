/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
  namespace oin = OpenKalman::internal;

  // ------------------- //
  //        Mean         //
  // ------------------- //

  /// A typed vector.
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, typed_matrix_nestable NestedMatrix> requires
    (RowCoefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct Mean : oin::TypedMatrixBase<
    Mean<RowCoefficients, NestedMatrix>, RowCoefficients, Axes<MatrixTraits<NestedMatrix>::columns>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<RowCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(RowCoefficients::dimensions == MatrixTraits<NestedMatrix>::rows);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this matrix.

  protected:

    using ColumnCoefficients = Axes<MatrixTraits<NestedMatrix>::columns>;

  private:

    using Base = oin::TypedMatrixBase<Mean, RowCoefficients, ColumnCoefficients, NestedMatrix>;

  public:

    using Base::Base;


    /// Construct from a compatible mean.
#ifdef __cpp_concepts
    template<mean Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } // \todo doesn't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<mean<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix{from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a compatible typed matrix or Euclidean-transformed mean.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and (not mean<Arg>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {wrap_angles<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not euclidean_transformed<Arg> and not mean<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {wrap_angles<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (MatrixTraits<Arg>::rows == MatrixTraits<NestedMatrix>::rows) and
      (MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns) and
      requires(Arg&& arg) { NestedMatrix {wrap_angles<RowCoefficients>(std::declval<Arg>())}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::rows == MatrixTraits<NestedMatrix>::rows) and
      (MatrixTraits<Arg>::columns == MatrixTraits<NestedMatrix>::columns) and
      std::is_constructible_v<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<Arg>()))>, int> = 0>
#endif
    explicit Mean(Arg&& arg) noexcept : Base {wrap_angles<RowCoefficients>(std::forward<Arg>(arg))} {}


    /// Construct from a list of coefficients.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { NestedMatrix {wrap_angles<RowCoefficients>(
        MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...))}; }
#else
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...> and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == MatrixTraits<NestedMatrix>::rows) or
        (sizeof...(Args) == MatrixTraits<NestedMatrix>::rows * MatrixTraits<NestedMatrix>::columns)) and
      std::is_constructible_v<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<NestedMatrix>()))>,
        int> = 0>
#endif
    Mean(Args ... args)
      : Base {wrap_angles<RowCoefficients>(MatrixTraits<NestedMatrix>::make(static_cast<const Scalar>(args)...))} {}


    /**
     * \brief Assign from a compatible \ref OpenKalman::mean "mean".
     */
#ifdef __cpp_concepts
    template<mean Arg> requires (not std::derived_from<std::decay_t<Arg>, Mean>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<mean<Arg> and (not std::is_base_of_v<Mean, std::decay_t<Arg>>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(nested_matrix(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::euclidean_transformed "euclidean_transformed".
     */
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_t<Arg>>()))>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(other))));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref OpenKalman::typed_matrix "typed_matrix" that is not
     * \ref OpenKalman::mean "mean" or \ref OpenKalman::euclidean_transformed "euclidean_transformed".
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not mean<Arg>) and (not euclidean_transformed<Arg>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and untyped_columns<Arg> and
      modifiable<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<nested_matrix_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not mean<Arg>) and (not euclidean_transformed<Arg>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and untyped_columns<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(wrap_angles<RowCoefficients>(nested_matrix(std::forward<Arg>(other))));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(wrap_angles<RowCoefficients>(std::forward<Arg>(arg)));
      }
      return *this;
    }


    /// Increment from another mean.
    auto& operator+=(const Mean& other)
    {
      if constexpr(RowCoefficients::axes_only)
        this->nested_matrix() += other.nested_matrix();
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(this->nested_matrix() + other.nested_matrix());
      return *this;
    }


    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (not euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      if constexpr(RowCoefficients::axes_only)
        this->nested_matrix() += nested_matrix(std::forward<Arg>(other));
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(
          this->nested_matrix() + nested_matrix(std::forward<Arg>(other)));
      return *this;
    }


    /// Add a stochastic value to each column of the mean, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){
        if constexpr(RowCoefficients::axes_only)
          col += arg().nested_matrix();
        else
          col = wrap_angles<RowCoefficients>(col + arg().nested_matrix());
      });
      return *this;
    }


    /// Decrement from another mean and wrap result.
    auto& operator-=(const Mean& other)
    {
      if constexpr(RowCoefficients::axes_only)
        this->nested_matrix() -= other.nested_matrix();
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(this->nested_matrix() - other.nested_matrix());
      return *this;
    }


    /// Decrement from another typed matrix and wrap result.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (not euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      if constexpr(RowCoefficients::axes_only)
        this->nested_matrix() -= nested_matrix(std::forward<Arg>(other));
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(
          this->nested_matrix() - nested_matrix(std::forward<Arg>(other)));
      return *this;
    }


    /// Subtract a stochastic value to each column of the mean, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){
        if constexpr(RowCoefficients::axes_only)
          col -= arg().nested_matrix();
        else
          col = wrap_angles<RowCoefficients>(col - arg().nested_matrix());
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
      if constexpr(RowCoefficients::axes_only)
        this->nested_matrix() *= s;
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(this->nested_matrix() * s);
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
      if constexpr(RowCoefficients::axes_only)
        this->nested_matrix() /= s;
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(this->nested_matrix() / s);
      return *this;
    }

  protected:

    template<typename C = RowCoefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return Mean<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  /// Deduce template parameters from a typed_matrix_nestable, assuming axis-only coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  explicit Mean(V&&) -> Mean<Axes<MatrixTraits<V>::rows>, passable_t<V>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \todo Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients, std::decay_t<decltype(
    wrap_angles<typename MatrixTraits<V>::RowCoefficients>(nested_matrix(std::forward<V>(std::declval<V>()))))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<typename MatrixTraits<V>::RowCoefficients, std::decay_t<decltype(
    from_euclidean<typename MatrixTraits<V>::RowCoefficients>(nested_matrix(std::forward<V>(std::declval<V>()))))>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a Mean from a typed_matrix_nestable, specifying the row coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable M> requires
    (Coefficients::dimensions == MatrixTraits<M>::rows)
#else
  template<typename Coefficients, typename M, std::enable_if_t<coefficients<Coefficients> and
    typed_matrix_nestable<M> and (Coefficients::dimensions == MatrixTraits<M>::rows), int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    constexpr auto rows = MatrixTraits<M>::rows;
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>, Axes<rows>, Coefficients>;
    decltype(auto) b = wrap_angles<Coeffs>(std::forward<M>(m)); using B = decltype(b);
    return Mean<Coeffs, passable_t<B>>(std::forward<B>(b));
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
    using Coeffs = Axes<MatrixTraits<M>::rows>;
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
    (MatrixTraits<M>::rows == Coefficients::dimensions)
#else
  template<typename Coefficients, typename M, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_nestable<M> and
    (MatrixTraits<M>::rows == Coefficients::dimensions), int> = 0>
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
    return make_mean<Axes<MatrixTraits<M>::rows>, M>();
  }


  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename Coeffs, typename NestedType>
  struct MatrixTraits<Mean<Coeffs, NestedType>>
  {
    using NestedMatrix = NestedType;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = MatrixTraits<NestedMatrix>::columns;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Axes<columns>;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static_assert(RowCoefficients::dimensions == rows);

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Mean<RowCoefficients, self_contained_t<NestedMatrix>>;


    /// Make from a typed_matrix_nestable. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
      (std::is_void_v<CC> or equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>) and
      (MatrixTraits<Arg>::rows == RC::dimensions)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      (std::is_void_v<CC> or equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>) and
      typed_matrix_nestable<Arg> and (MatrixTraits<Arg>::rows == RC::dimensions), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      decltype(auto) b = wrap_angles<RC>(std::forward<Arg>(arg)); using B = decltype(b);
      return Mean<RC, std::decay_t<B>>(std::forward<B>(b));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires
    (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0) + (columns == 0 ? 1 : 0)), int> = 0>
#endif
    static auto zero(const Args...args)
    {
      return make(MatrixTraits<NestedMatrix>::zero(static_cast<std::size_t>(args)...));
    }


#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t> ... Args> requires (sizeof...(Args) == (rows == 0 ? 1 : 0))
#else
    template<typename...Args, std::enable_if_t<(std::is_convertible_v<Args, std::size_t> and ...) and
      (sizeof...(Args) == (rows == 0 ? 1 : 0)), int> = 0>
#endif
    static auto identity(const Args...args)
    {
      auto b = MatrixTraits<NestedMatrix>::identity(args...);
      return Matrix<RowCoefficients, RowCoefficients, decltype(b)>(std::move(b));
    }

  };


} // namespace OpenKalman


#endif //OPENKALMAN_MEAN_HPP
