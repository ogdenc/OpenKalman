/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SQUAREROOTCOVARIANCE_HPP
#define OPENKALMAN_SQUAREROOTCOVARIANCE_HPP


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // ---------------------- //
  //  SquareRootCovariance  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, covariance_nestable NestedMatrix> requires
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and scalar_type<scalar_type_of_t<NestedMatrix>>
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct SquareRootCovariance : oin::CovarianceImpl<SquareRootCovariance<TypedIndex, NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(fixed_index_descriptor<TypedIndex>);
    static_assert(covariance_nestable<NestedMatrix>);
    static_assert(dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(scalar_type<scalar_type_of_t<NestedMatrix>>);
#endif

    // May be accessed externally through MatrixTraits:
    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  private:

    using Base = oin::CovarianceImpl<SquareRootCovariance, NestedMatrix>;
    using typename Base::CholeskyNestedMatrix;
    using Base::nested_matrix;
    using Base::cholesky_nested_matrix;
    using Base::synchronization_direction;
    using Base::synchronize_forward;
    using Base::synchronize_reverse;
    using Base::mark_nested_matrix_changed;
    using Base::mark_cholesky_nested_matrix_changed;
    using Base::mark_synchronized;


    // May be accessed externally through MatrixTraits:
    static constexpr auto dim = row_dimension_of_v<NestedMatrix>;

    // May be accessed externally through MatrixTraits:
    static constexpr TriangleType triangle_type =
      triangle_type_of_v<typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<>>;

    // A triangular nested matrix type.
    using NestedTriangular = std::conditional_t<triangular_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<triangle_type>>;


    // A function that makes a self-contained covariance from a nested matrix.
    template<typename C = TypedIndex, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


    /**
     * \brief Construct from a non-square-root, non-diagonal \ref covariance.
     */
#ifdef __cpp_concepts
    template<self_adjoint_covariance M> requires (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (hermitian_matrix<nested_matrix_of_t<M>> == hermitian_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<self_adjoint_covariance<M> and
      (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (hermitian_matrix<nested_matrix_of_t<M>> == hermitian_matrix<NestedMatrix>), int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


  public:
    // ------------ //
    // Constructors //
    // ------------ //

    /// Default constructor.
#ifdef __cpp_concepts
    SquareRootCovariance() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    SquareRootCovariance()
#endif
      : Base {} {}


    /**
     * \brief Construct from another \ref triangular_covariance.
     */
#ifdef __cpp_concepts
    template<triangular_covariance M> requires (not std::derived_from<std::decay_t<M>, SquareRootCovariance>) and
      (triangle_type_of_v<M> == triangle_type_of_v<SquareRootCovariance>) and requires(M&& m) { Base {std::forward<M>(m)}; }
#else
    template<typename M, std::enable_if_t<
      triangular_covariance<M> and (not std::is_base_of_v<SquareRootCovariance, std::decay_t<M>>) and
      (triangle_type_of<M>::value == triangle_type_of<SquareRootCovariance>::value) and
      std::is_constructible_v<Base, M&&>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable M> requires requires(M&& m) { Base {std::forward<M>(m)}; }
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      std::is_constructible_v<Base, M&&>, int> = 0>
#endif
    explicit SquareRootCovariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref typed_matrix.
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be triangular, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix M> requires (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<row_coefficient_types_of_t<M>, TypedIndex> and
      requires(M&& m) { Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))}; }
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<row_coefficient_types_of_t<M>, TypedIndex> and
      std::is_constructible_v<Base,
        decltype(oin::to_covariance_nestable<NestedTriangular>(std::declval<M&&>()))>, int> = 0>
#endif
    explicit SquareRootCovariance(M&& m) noexcept
      : Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a \ref typed_matrix_nestable.
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be triangular, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      requires(M&& m) { Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))}; }
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      std::is_constructible_v<Base,
        decltype(oin::to_covariance_nestable<NestedTriangular>(std::declval<M&&>()))>, int> = 0>
#endif
    explicit SquareRootCovariance(M&& m) noexcept
      : Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))} {}


    /// Construct from Scalar coefficients. Assumes matrix is triangular, and only reads lower left triangle.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { Base {MatrixTraits<std::decay_t<NestedTriangular>>::make(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, const Scalar> and ...) and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == dim) or
        (sizeof...(Args) == dim * dim)) and std::is_constructible_v<Base, NestedTriangular&&>, int> = 0>
#endif
    SquareRootCovariance(Args ... args)
      : Base {MatrixTraits<std::decay_t<NestedTriangular>>::make(static_cast<const Scalar>(args)...)} {}


    // ---------------------- //
    //  Assignment Operators  //
    // ---------------------- //

    /**
     * \brief Assign from a compatible \ref triangular_covariance.
     * \note the triangle types must match.
     */
#ifdef __cpp_concepts
    template<triangular_covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, SquareRootCovariance>) and
      (triangle_type_of_v<Arg> == triangle_type_of_v<SquareRootCovariance>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<triangular_covariance<Arg> and
      (not std::is_base_of_v<SquareRootCovariance, std::decay_t<Arg>>) and
      (triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(other));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref typed_matrix (assumed, without checking, to be triangular).
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires square_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, NestedTriangular>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, NestedTriangular>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(oin::to_covariance_nestable<NestedTriangular>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and modifiable<NestedMatrix, Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(other));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref typed_matrix_nestable (assumed, without checking, to be triangular).
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and square_matrix<Arg> and
      modifiable<NestedMatrix, NestedTriangular>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
      square_matrix<Arg> and modifiable<NestedMatrix, NestedTriangular>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(oin::to_covariance_nestable<NestedTriangular>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * Increment by another \ref triangular_covariance or triangular \ref typed_matrix.
     * \warning This is computationally expensive if the nested matrix is not \ref triangular_matrix.
     * This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((triangular_covariance<Arg> and triangle_type_of_v<Arg> == triangle_type_of_v<SquareRootCovariance>) or
        (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((triangular_covariance<Arg> and triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) or
        (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>, int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      if constexpr(triangular_matrix<NestedMatrix>) // Case 1 or 2
      {
        nested_matrix() += oin::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else // Case 3 or 4
      {
        if (synchronization_direction() > 0) synchronize_forward();
        cholesky_nested_matrix() += oin::to_covariance_nestable<NestedTriangular>(arg);
        if (synchronization_direction() > 0)
        {
          Base::synchronize_reverse();
        }
        else
        {
          mark_cholesky_nested_matrix_changed();
        }
      }
      return *this;
    }


    /**
     * Increment by another SquareRootCovariance of the same type.
     * \warning This is computationally expensive if the nested matrix is self-adjoint. This can generally be avoided.
     */
#ifdef __cpp_concepts
    auto& operator+=(const SquareRootCovariance& arg)
      requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>), int> = 0>
    auto& operator+=(const SquareRootCovariance& arg)
#endif
    {
      return operator+=<const SquareRootCovariance&>(arg);
    }


    /**
     * Decrement by another \ref triangular_covariance or triangular \ref typed_matrix.
     * \warning This is computationally expensive if the nested matrix is not \ref triangular_matrix.
     * This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
    ((triangular_covariance<Arg> and triangle_type_of_v<Arg> == triangle_type_of_v<SquareRootCovariance>) or
      (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((triangular_covariance<Arg> and triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) or
        (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() -= oin::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        cholesky_nested_matrix() -= oin::to_covariance_nestable<NestedTriangular>(arg);
        if (synchronization_direction() > 0)
        {
          synchronize_reverse();
        }
        else
        {
          mark_cholesky_nested_matrix_changed();
        }
      }
      return *this;
    }


    /**
     * Decrement by another SquareRootCovariance of the same type.
     * \warning This is computationally expensive if the nested matrix is self-adjoint. This can generally be avoided.
     */
#ifdef __cpp_concepts
    auto& operator-=(const SquareRootCovariance& arg)
      requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>), int> = 0>
    auto& operator-=(const SquareRootCovariance& arg)
#endif
    {
      return operator-=<const SquareRootCovariance&>(arg);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() *= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0) nested_matrix() *= static_cast<const Scalar>(s) * s;
        if (synchronization_direction() <= 0) cholesky_nested_matrix() *= static_cast<const Scalar>(s);
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator/=(const S s)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() /= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0) nested_matrix() /= static_cast<const Scalar>(s) * s;
        if (synchronization_direction() <= 0) cholesky_nested_matrix() /= static_cast<const Scalar>(s);
      }
      return *this;
    }


    /**
     * \brief Multiply by another \ref triangular_covariance.
     * \details The underlying triangle type (upper or lower) of Arg much match that of the nested matrix.
     * \warning This is computationally expensive unless the nested matrices of *this and Arg are both either
     * triangular or self-adjoint.
     */
#ifdef __cpp_concepts
    template<triangular_covariance Arg> requires (triangle_type_of_v<Arg> == triangle_type_of_v<SquareRootCovariance>) and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename Arg, std::enable_if_t<triangular_covariance<Arg> and
      (triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator*=(const Arg& arg)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() *= oin::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        cholesky_nested_matrix() *= oin::to_covariance_nestable<NestedTriangular>(arg);
        if (synchronization_direction() > 0)
        {
          synchronize_reverse();
        }
        else
        {
          mark_cholesky_nested_matrix_changed();
        }
      }
      return *this;
    }


    // ------- //
    //  Other  //
    // ------- //

    /**
     * \brief Take the Cholesky square of *this.
     * \details If *this is an lvalue reference, this creates a reference to the nested matrix rather than a copy.
     * \return A Covariance based on *this.
     * \note One cannot assume that the lifetime of the result is longer than the lifetime of the object.
     */
    auto square() &
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return Covariance<TypedIndex, std::remove_reference_t<NestedMatrix>&> {*this};
      }
      else
      {
        auto n = Cholesky_square(nested_matrix());
        return Covariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square() const &
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return Covariance<TypedIndex, const std::remove_reference_t<NestedMatrix>&> {*this};
      }
      else
      {
        auto n = Cholesky_square(nested_matrix());
        return Covariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square() &&
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return Covariance<TypedIndex, std::remove_reference_t<NestedMatrix>> {std::move(*this)};
      }
      else
      {
        auto n = Cholesky_square(std::move(*this).nested_matrix());
        return Covariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square() const &&
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return Covariance<TypedIndex, std::remove_reference_t<NestedMatrix>> {std::move(*this)};
      }
      else
      {
        auto n = Cholesky_square(std::move(*this).nested_matrix());
        return Covariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /**
     * \brief Perform a rank update.
     */
#ifdef __cpp_concepts
    template<typed_matrix U> requires equivalent_to<row_coefficient_types_of_t<U>, TypedIndex> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      equivalent_to<row_coefficient_types_of_t<U>, TypedIndex> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& rank_update(const U& u, const Scalar alpha = 1) &
    {
      if (synchronization_direction() < 0) synchronize_reverse();
      OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha);
      mark_nested_matrix_changed();
      return *this;
    }


    /**
     * \brief Perform a rank update.
     */
#ifdef __cpp_concepts
    template<typed_matrix U> requires equivalent_to<row_coefficient_types_of_t<U>, TypedIndex>
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      equivalent_to<row_coefficient_types_of_t<U>, TypedIndex>, int> = 0>
#endif
    auto rank_update(const U& u, const Scalar alpha = 1) &&
    {
      if (synchronization_direction() < 0) synchronize_reverse();
      return make(OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha));
    }

  private:

#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct oin::CovarianceBase;


    template<typename, typename>
    friend struct oin::CovarianceImpl;


    template<typename, typename>
    friend struct oin::CovarianceBase3Impl;


#ifdef __cpp_concepts
    template<fixed_index_descriptor C, covariance_nestable N> requires
    (dimension_size_of_v<C> == row_dimension_of_v<N>) and (not std::is_rvalue_reference_v<N>)
#else
    template<typename, typename>
#endif
    friend struct SquareRootCovariance;


#ifdef __cpp_concepts
    template<fixed_index_descriptor C, covariance_nestable N> requires
      (dimension_size_of_v<C> == row_dimension_of_v<N>) and (not std::is_rvalue_reference_v<N>)
#else
    template<typename, typename>
#endif
    friend struct Covariance;

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  /**
   * \brief Deduce SquareRootCovariance type from a \ref covariance_nestable.
   */
#ifdef __cpp_concepts
  template<covariance_nestable M>
#else
  template<typename M, std::enable_if_t<covariance_nestable<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<Dimensions<row_dimension_of_v<M>>, passable_t<M>>;


  /**
   * \brief Deduce SquareRootCovariance type from a square \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix M> requires square_matrix<M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M> and square_matrix<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<row_coefficient_types_of_t<M>,
    typename MatrixTraits<std::decay_t<nested_matrix_of_t<M>>>::template TriangularMatrixFrom<>>;


  /**
   * \brief Deduce SquareRootCovariance type from a square \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and square_matrix<M>
#else
  template<typename M, std::enable_if_t<
    typed_matrix_nestable<M> and (not covariance_nestable<M>) and square_matrix<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<Dimensions<row_dimension_of_v<M>>,
    typename MatrixTraits<std::decay_t<M>>::template TriangularMatrixFrom<>>;


  // ---------------- //
  //  Make Functions  //
  // ---------------- //

  /**
   * \brief Make a SquareRootCovariance from a \ref covariance_nestable, specifying the coefficients.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable with size matching TypedIndex.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, covariance_nestable Arg> requires
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<Arg>)
#else
  template<typename TypedIndex, typename Arg, std::enable_if_t<fixed_index_descriptor<TypedIndex> and
    covariance_nestable<Arg> and (dimension_size_of_v<TypedIndex> == row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    return SquareRootCovariance<TypedIndex, passable_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance from a \ref covariance_nestable, with default Axis coefficients.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable.
   */
#ifdef __cpp_concepts
  template<covariance_nestable Arg>
#else
  template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_square_root_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance (with nested triangular matrix) from a self-adjoint \ref typed_matrix_nestable.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable with size matching TypedIndex.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, TriangleType triangle_type = TriangleType::lower, typed_matrix_nestable Arg>
  requires (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<Arg>) and (dimension_size_of_v<TypedIndex> == column_dimension_of_v<Arg>)
#else
  template<typename TypedIndex, TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<
    fixed_index_descriptor<TypedIndex> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (dimension_size_of_v<TypedIndex> == row_dimension_of<Arg>::value) and
    (dimension_size_of_v<TypedIndex> == column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using T = typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<triangle_type>;
    return SquareRootCovariance<TypedIndex, T>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a default Axis SquareRootCovariance from a self-adjoint \ref typed_matrix_nestable.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type = TriangleType::lower, typed_matrix_nestable Arg> requires
    (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>
#else
  template<TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<
    typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_square_root_covariance<C, triangle_type>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, TriangleType triangle_type, typed_matrix_nestable Arg> requires square_matrix<Arg>
#else
  template<typename TypedIndex, TriangleType triangle_type, typename Arg,
    std::enable_if_t<fixed_index_descriptor<TypedIndex> and typed_matrix_nestable<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using B = std::conditional_t<triangle_type == TriangleType::diagonal,
    typename MatrixTraits<std::decay_t<Arg>>::template DiagonalMatrixFrom<>,
      typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<triangle_type>>;
    return SquareRootCovariance<TypedIndex, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref covariance_nestable or \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typename Arg> requires
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename TypedIndex, typename Arg, std::enable_if_t<
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    constexpr TriangleType template_type = triangle_type_of_v<typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<>>;
    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<std::decay_t<Arg>>::template DiagonalMatrixFrom<>,
      std::conditional_t<hermitian_matrix<Arg>,
        typename MatrixTraits<std::decay_t<Arg>>::template SelfAdjointMatrixFrom<template_type>,
        typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<template_type>>>;
    return SquareRootCovariance<TypedIndex, B>();
  }


/**
 * \overload
 * \brief Make a writable, uninitialized SquareRootCovariance from a \ref typed_matrix_nestable or \ref covariance_nestable.
 * \details The coefficients will be Axis.
 */
#ifdef __cpp_concepts
  template<typename Arg> requires (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<(covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_square_root_covariance<C, Arg>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix_nestable Arg> requires square_matrix<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<
    typed_matrix_nestable<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_square_root_covariance<C, triangle_type, Arg>();
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance based on another \ref triangular_covariance.
   */
#ifdef __cpp_concepts
  template<triangular_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<triangular_covariance<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = row_coefficient_types_of_t<Arg>;
    return SquareRootCovariance<C, nested_matrix_of_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref triangular_covariance.
   */
#ifdef __cpp_concepts
  template<triangular_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<triangular_covariance<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = row_coefficient_types_of_t<Arg>;
    using B = nested_matrix_of_t<Arg>;
    return make_square_root_covariance<C, B>();
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance from a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type = TriangleType::lower, typed_matrix Arg> requires
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>
#else
  template<TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = row_coefficient_types_of_t<Arg>;
    return make_square_root_covariance<C, triangle_type>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance based on a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg> requires square_matrix<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<
    typed_matrix<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = row_coefficient_types_of_t<Arg>;
    using B = nested_matrix_of_t<Arg>;
    return make_square_root_covariance<C, triangle_type, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance based on a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = row_coefficient_types_of_t<Arg>;
    using B = nested_matrix_of_t<Arg>;
    return make_square_root_covariance<C, B>();
  }


}


#endif //OPENKALMAN_SQUAREROOTCOVARIANCE_HPP

