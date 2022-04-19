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
  template<typed_index_descriptor RowCoefficients, typed_matrix_nestable NestedMatrix> requires
    (dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct Mean : oin::TypedMatrixBase<
    Mean<RowCoefficients, NestedMatrix>, NestedMatrix, RowCoefficients, Dimensions<column_dimension_of_v<NestedMatrix>>>
  {

#ifndef __cpp_concepts
    static_assert(typed_index_descriptor<RowCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  protected:

    using ColumnCoefficients = Dimensions<column_dimension_of_v<NestedMatrix>>;

  private:

    using Base = oin::TypedMatrixBase<Mean, NestedMatrix, RowCoefficients, ColumnCoefficients>;

  public:

    using Base::Base;


    /// Construct from a compatible mean.
#ifdef __cpp_concepts
    template<mean Arg> requires
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      //requires(Arg&& arg) { NestedMatrix {nested_matrix(std::forward<Arg>(arg))}; } // \todo doesn't work in GCC 10
      std::constructible_from<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>
#else
    template<typename Arg, std::enable_if_t<mean<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix, decltype(nested_matrix(std::declval<Arg&&>()))>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {nested_matrix(std::forward<Arg>(arg))} {}


    /// Construct from a compatible Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix{from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      std::is_constructible_v<NestedMatrix,
        decltype(from_euclidean<RowCoefficients>(nested_matrix(std::declval<Arg&&>())))>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {from_euclidean<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a compatible typed matrix or Euclidean-transformed mean.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and (not mean<Arg>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      requires(Arg&& arg) { NestedMatrix {wrap_angles<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and not euclidean_transformed<Arg> and not mean<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients>, int> = 0>
#endif
    Mean(Arg&& arg) noexcept : Base {wrap_angles<RowCoefficients>(nested_matrix(std::forward<Arg>(arg)))} {}


    /// Construct from a typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (row_dimension_of_v<Arg> == row_dimension_of_v<NestedMatrix>) and
      (column_dimension_of_v<Arg> == column_dimension_of_v<NestedMatrix>) and
      requires(Arg&& arg) { NestedMatrix {wrap_angles<RowCoefficients>(std::declval<Arg>())}; }
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (row_dimension_of<Arg>::value == row_dimension_of<NestedMatrix>::value) and
      (column_dimension_of<Arg>::value == column_dimension_of<NestedMatrix>::value) and
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
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == row_dimension_of<NestedMatrix>::value) or
        (sizeof...(Args) == row_dimension_of<NestedMatrix>::value * column_dimension_of<NestedMatrix>::value)) and
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
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<mean<Arg> and (not std::is_base_of_v<Mean, std::decay_t<Arg>>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
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
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      modifiable<NestedMatrix, decltype(from_euclidean<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>, int> = 0>
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
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and has_untyped_index<Arg, 1> and
      modifiable<NestedMatrix, decltype(wrap_angles<RowCoefficients>(std::declval<nested_matrix_of_t<Arg>>()))>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not mean<Arg>) and (not euclidean_transformed<Arg>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and has_untyped_index<Arg, 1>, int> = 0>
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
      if constexpr(untyped_index_descriptor<RowCoefficients>)
        this->nested_matrix() += other.nested_matrix();
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(this->nested_matrix() + other.nested_matrix());
      return *this;
    }


    /// Increment from another typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      if constexpr(untyped_index_descriptor<RowCoefficients>)
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
      apply_columnwise([&arg](auto& col){
        if constexpr(untyped_index_descriptor<RowCoefficients>)
          col += arg().nested_matrix();
        else
          col = wrap_angles<RowCoefficients>(col + arg().nested_matrix());
      }, this->nested_matrix());
      return *this;
    }


    /// Decrement from another mean and wrap result.
    auto& operator-=(const Mean& other)
    {
      if constexpr(untyped_index_descriptor<RowCoefficients>)
        this->nested_matrix() -= other.nested_matrix();
      else
        this->nested_matrix() = wrap_angles<RowCoefficients>(this->nested_matrix() - other.nested_matrix());
      return *this;
    }


    /// Decrement from another typed matrix and wrap result.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, RowCoefficients> and
      equivalent_to<column_coefficient_types_of_t<Arg>, ColumnCoefficients> and
      (not euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      if constexpr(untyped_index_descriptor<RowCoefficients>)
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
      apply_columnwise([&arg](auto& col){
        if constexpr(untyped_index_descriptor<RowCoefficients>)
          col -= arg().nested_matrix();
        else
          col = wrap_angles<RowCoefficients>(col - arg().nested_matrix());
      }, this->nested_matrix());
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
      if constexpr(untyped_index_descriptor<RowCoefficients>)
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
      if constexpr(untyped_index_descriptor<RowCoefficients>)
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

  /// Deduce template parameters from a typed_matrix_nestable, assuming untyped index descriptors.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  explicit Mean(V&&) -> Mean<Dimensions<row_dimension_of_v<V>>, passable_t<V>>;


  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<row_coefficient_types_of_t<V>, std::decay_t<decltype(
    wrap_angles<row_coefficient_types_of_t<V>>(nested_matrix(std::forward<V>(std::declval<V>()))))>>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS_2
  // \todo Unlike SFINAE version, this incorrectly matches V==Mean and V==Matrix in both GCC 10.1.0 and clang 10.0.0:
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<euclidean_transformed<V>, int> = 0>
#endif
  Mean(V&&) -> Mean<row_coefficient_types_of_t<V>, std::decay_t<decltype(
    from_euclidean<row_coefficient_types_of_t<V>>(nested_matrix(std::forward<V>(std::declval<V>()))))>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a Mean from a typed_matrix_nestable, specifying the row typed_index_descriptor.
   * \tparam Coefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<typed_index_descriptor Coefficients, typed_matrix_nestable M> requires
    (dimension_size_of_v<Coefficients> == row_dimension_of_v<M>)
#else
  template<typename Coefficients, typename M, std::enable_if_t<typed_index_descriptor<Coefficients> and
    typed_matrix_nestable<M> and (dimension_size_of_v<Coefficients> == row_dimension_of<M>::value), int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    constexpr auto rows = row_dimension_of_v<M>;
    using Coeffs = std::conditional_t<std::is_void_v<Coefficients>, Dimensions<rows>, Coefficients>;
    decltype(auto) b = wrap_angles<Coeffs>(std::forward<M>(m)); using B = decltype(b);
    return Mean<Coeffs, passable_t<B>>(std::forward<B>(b));
  }


  /**
   * \overload
   * \brief Make a Mean from a typed_matrix_nestable object, with default Axis typed_index_descriptor.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  inline auto make_mean(M&& m) noexcept
  {
    using Coeffs = Dimensions<row_dimension_of_v<M>>;
    return make_mean<Coeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a Mean from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires has_untyped_index<Arg, 1>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1>, int> = 0>
#endif
  inline auto make_mean(Arg&& arg) noexcept
  {
    using C = row_coefficient_types_of_t<Arg>;
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
  template<typed_index_descriptor Coefficients, typed_matrix_nestable M> requires
    (row_dimension_of_v<M> == dimension_size_of_v<Coefficients>)
#else
  template<typename Coefficients, typename M, std::enable_if_t<
    typed_index_descriptor<Coefficients> and typed_matrix_nestable<M> and
    (row_dimension_of<M>::value == dimension_size_of_v<Coefficients>), int> = 0>
#endif
  inline auto make_mean()
  {
    return Mean<Coefficients, dense_writable_matrix_t<M>>();
  }


  /**
   * \overload
   * \brief Make a self-contained Mean with default Axis typed_index_descriptor.
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
    return make_mean<Dimensions<row_dimension_of_v<M>>, M>();
  }


} // namespace OpenKalman


#endif //OPENKALMAN_MEAN_HPP
