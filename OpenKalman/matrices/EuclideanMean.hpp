/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
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
  template<coefficients RowCoefficients, typed_matrix_nestable NestedMatrix> requires
    (RowCoefficients::euclidean_dimension == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct EuclideanMean : internal::TypedMatrixBase<EuclideanMean<RowCoefficients, NestedMatrix>,
    RowCoefficients, Axes<MatrixTraits<NestedMatrix>::columns>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<RowCoefficients>);
    static_assert(typed_matrix_nestable<NestedMatrix>);
    static_assert(RowCoefficients::euclidean_dimension == MatrixTraits<NestedMatrix>::dimension);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this matrix.

  protected:

    using ColumnCoefficients = Axes<MatrixTraits<NestedMatrix>::columns>;

  private:

    using Base = internal::TypedMatrixBase<EuclideanMean, RowCoefficients, ColumnCoefficients, NestedMatrix>;

  public:

    using Base::Base;


    /// Copy constructor.
    EuclideanMean(const EuclideanMean& other) : Base(other.nested_matrix()) {}


    /// Move constructor.
    EuclideanMean(EuclideanMean&& other) noexcept : Base(std::move(other).nested_matrix()) {}


    /// Construct from a compatible Euclidean-transformed matrix.
#ifdef __cpp_concepts
    template<euclidean_transformed Arg> requires
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>) and
      (equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>)
#else
    template<typename Arg, std::enable_if_t<euclidean_transformed<Arg> and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>) and
      (equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>), int> = 0>
#endif
    EuclideanMean(Arg&& other) noexcept : Base(std::forward<Arg>(other).nested_matrix()) {}


    /// Construct from a compatible non-Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not euclidean_transformed<Arg>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>) and
      (equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and (not euclidean_transformed<Arg>) and
      (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients>) and
      (equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients>), int> = 0>
#endif
    EuclideanMean(Arg&& other) noexcept
      : Base(to_euclidean<RowCoefficients>(std::forward<Arg>(other).nested_matrix())) {}


    /// Construct from compatible typed matrix object.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (MatrixTraits<Arg>::dimension == Base::dimension) and
      (MatrixTraits<Arg>::columns == Base::columns) and std::is_constructible_v<Base, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::dimension == Base::dimension) and (MatrixTraits<Arg>::columns == Base::columns) and
      std::is_constructible_v<Base, Arg>, int> = 0>
#endif
    EuclideanMean(Arg&& arg) noexcept : Base(std::forward<Arg>(arg)) {}


    /// Copy assignment operator.
    auto& operator=(const EuclideanMean& other)
    {
      Base::operator=(other);
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(EuclideanMean&& other)
    {
      Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from a compatible \ref typed_matrix, where no transformation to Euclidean space is required.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, EuclideanMean>) and
      (euclidean_transformed<Arg> or RowCoefficients::axes_only) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      untyped_columns<Arg> and modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not std::is_base_of_v<EuclideanMean, std::decay_t<Arg>>) and
      (euclidean_transformed<Arg> or RowCoefficients::axes_only) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      untyped_columns<Arg> and modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(other).nested_matrix());
      }
      return *this;
    }


    /// Assign from a compatible \ref typed_matrix, where a transformation to Euclidean space is required.
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires (not std::derived_from<std::decay_t<Arg>, EuclideanMean>) and
      (not euclidean_transformed<Arg> and not RowCoefficients::axes_only) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      untyped_columns<Arg> and modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      (not std::is_base_of_v<EuclideanMean, std::decay_t<Arg>>) and
      (not euclidean_transformed<Arg> and not RowCoefficients::axes_only) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      untyped_columns<Arg> and modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(to_euclidean<RowCoefficients>(std::forward<Arg>(other).nested_matrix()));
      }
      return *this;
    }


    /// Assign from a compatible \ref typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
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
    template<typed_matrix Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (RowCoefficients::axes_only or euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (RowCoefficients::axes_only or euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator+=(Arg&& other) noexcept
    {
      this->nested_matrix() += std::forward<Arg>(other).nested_matrix();
      return *this;
    }


    /// Add a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (RowCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (RowCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>), int> = 0>
#endif
    auto& operator+=(const Arg& arg) noexcept
    {
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
    template<typed_matrix Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (RowCoefficients::axes_only or euclidean_transformed<Arg>)
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, RowCoefficients> and
      equivalent_to<typename MatrixTraits<Arg>::ColumnCoefficients, ColumnCoefficients> and
      (RowCoefficients::axes_only or euclidean_transformed<Arg>), int> = 0>
#endif
    auto& operator-=(Arg&& other) noexcept
    {
      this->nested_matrix() -= std::forward<Arg>(other).nested_matrix();
      return *this;
    }


    /// Subtract a stochastic value to each column of the matrix, based on a distribution.
#ifdef __cpp_concepts
    template<distribution Arg> requires (RowCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>)
#else
    template<typename Arg, std::enable_if_t<distribution<Arg> and (RowCoefficients::axes_only) and
      (equivalent_to<typename DistributionTraits<Arg>::Coefficients, RowCoefficients>), int> = 0>
#endif
    auto& operator-=(const Arg& arg) noexcept
    {
      apply_columnwise(this->nested_matrix(), [&arg](auto& col){ col -= arg().nested_matrix(); });
      return *this;
    }

  protected:

    template<typename C = RowCoefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return EuclideanMean<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

  public:

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  /// Deduce template parameters from a non-Euclidean-transformed typed matrix.
#if defined(__cpp_concepts) and false
  // \todo Unlike SFINAE version, this incorrectly matches V==EuclideanMean in both GCC 10.1.0 and clang 10.0.0:
  template<typed_matrix V> requires (not euclidean_transformed<V>) and untyped_columns<V> and
    (MatrixTraits<V>::RowCoefficients::euclidean_dimension == MatrixTraits<V>::dimension)
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and not euclidean_transformed<V> and untyped_columns<V> and
    MatrixTraits<V>::RowCoefficients::euclidean_dimension == MatrixTraits<V>::dimension, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients, decltype(
      to_euclidean<typename MatrixTraits<V>::RowCoefficients>(std::forward<V>(std::declval<V>()).nested_matrix()))>;


  /// Deduce template parameters from a Euclidean-transformed typed matrix.
#ifdef __cpp_concepts
  template<euclidean_transformed V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V> and euclidean_transformed<V>, int> = 0>
#endif
  EuclideanMean(V&&) -> EuclideanMean<typename MatrixTraits<V>::RowCoefficients, nested_matrix_t<V>>;


  /// Deduce template parameters from a typed_matrix_nestable, assuming axis-only coefficients.
#ifdef __cpp_concepts
  template<typed_matrix_nestable V>
#else
  template<typename V, std::enable_if_t<typed_matrix_nestable<V>, int> = 0>
#endif
  explicit EuclideanMean(V&&) -> EuclideanMean<Axes<MatrixTraits<V>::dimension>, passable_t<V>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

  /**
   * \brief Make a EuclideanMean from a typed_matrix_nestable, specifying the row coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows.
   * \tparam M A typed_matrix_nestable with size matching ColumnCoefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable M> requires
    (Coefficients::euclidean_dimension == MatrixTraits<M>::dimension)
#else
  template<typename Coefficients, typename M, std::enable_if_t<coefficients<Coefficients> and
    typed_matrix_nestable<M> and (Coefficients::euclidean_dimension == MatrixTraits<M>::dimension), int> = 0>
#endif
  auto make_euclidean_mean(M&& arg) noexcept
  {
    return EuclideanMean<Coefficients, passable_t<M>>(std::forward<M>(arg));
  }


  /**
   * \overload
   * \brief Make a EuclideanMean from a typed_matrix_nestable object, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  auto make_euclidean_mean(M&& m) noexcept
  {
    using Coeffs = Axes<MatrixTraits<M>::dimension>;
    return make_mean<Coeffs>(std::forward<M>(m));
  }


  /**
   * \overload
   * \brief Make a EuclideanMean from another typed_matrix.
   * \tparam Arg A typed_matrix (i.e., Matrix, Mean, or EuclideanMean).
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires untyped_columns<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and untyped_columns<Arg>, int> = 0>
#endif
  inline auto make_euclidean_mean(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(euclidean_transformed<Arg>)
      return make_euclidean_mean<C>(nested_matrix(std::forward<Arg>(arg)));
    else
      return make_euclidean_mean<C>(nested_matrix(to_euclidean<C>(std::forward<Arg>(arg))));
  }


  /**
   * \overload
   * \brief Make a default, self-contained EuclideanMean.
   * \tparam Coefficients The coefficient types corresponding to the rows.
   * \tparam M a typed_matrix_nestable on which the new matrix is based. It will be converted to a self_contained type
   * if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable M> requires
    (Coefficients::euclidean_dimension == MatrixTraits<M>::dimension)
#else
  template<typename Coefficients, typename M, std::enable_if_t<coefficients<Coefficients> and
    typed_matrix_nestable<M> and (Coefficients::euclidean_dimension == MatrixTraits<M>::dimension), int> = 0>
#endif
  auto make_euclidean_mean()
  {
    constexpr auto rows = MatrixTraits<M>::dimension;
    return EuclideanMean<Coefficients, native_matrix_t<M, rows>>();
  }


  /**
   * \overload
   * \brief Make a self-contained EuclideanMean with default Axis coefficients.
   * \tparam M a typed_matrix_nestable on which the new Euclidean mean is based.
   * It will be converted to a self_contained type if it is not already self-contained.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M>
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M>, int> = 0>
#endif
  auto make_euclidean_mean()
  {
    return make_euclidean_mean<Axes<MatrixTraits<M>::dimension>, M>();
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
    static_assert(RowCoefficients::euclidean_dimension == dimension);

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, rows, cols, S>;

    using SelfContainedFrom = EuclideanMean<RowCoefficients, self_contained_t<NestedMatrix>>;


    /// Make from a regular matrix. If CC is specified, it must be axes-only.
#ifdef __cpp_concepts
    template<coefficients RC = RowCoefficients, typename CC = void, typed_matrix_nestable Arg> requires
    (std::is_void_v<CC> or equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>) and
    (MatrixTraits<Arg>::dimension == RC::euclidean_dimension)
#else
    template<typename RC = RowCoefficients, typename CC = void, typename Arg, std::enable_if_t<
      coefficients<RC> and (std::is_void_v<CC> or equivalent_to<CC, Axes<MatrixTraits<Arg>::columns>>) and
      typed_matrix_nestable<Arg> and (MatrixTraits<Arg>::dimension == RC::euclidean_dimension), int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
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
