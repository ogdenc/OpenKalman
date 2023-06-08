/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCE_HPP
#define OPENKALMAN_COVARIANCE_HPP


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // ------------ //
  //  Covariance  //
  // ------------ //

#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, covariance_nestable NestedMatrix> requires
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and scalar_type<scalar_type_of_t<NestedMatrix>>
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct Covariance : oin::CovarianceImpl<Covariance<TypedIndex, NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(fixed_index_descriptor<TypedIndex>);
    static_assert(covariance_nestable<NestedMatrix>);
    static_assert(dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(scalar_type<scalar_type_of_t<NestedMatrix>>);
#endif

    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  private:

    using Base = oin::CovarianceImpl<Covariance, NestedMatrix>;
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
    static constexpr TriangleType storage_triangle =
      triangle_type_of_v<typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<>>;

    // A self-adjoint nested matrix type.
    using NestedSelfAdjoint = std::conditional_t<hermitian_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<std::decay_t<NestedMatrix>>::template SelfAdjointMatrixFrom<storage_triangle>>;


    // A function that makes a covariance from a nested matrix.
    template<typename C = TypedIndex, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


    /**
     * \brief Construct from a non-diagonal \ref triangular_covariance.
     */
#ifdef __cpp_concepts
    template<triangular_covariance M> requires (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (hermitian_matrix<nested_matrix_of_t<M>> == hermitian_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<triangular_covariance<M> and
      (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (hermitian_matrix<nested_matrix_of_t<M>> == hermitian_matrix<NestedMatrix>), int> = 0>
#endif
    Covariance(M&& m) noexcept : Base {std::forward<M>(m)} {}

  public:

    // -------------- //
    //  Constructors  //
    // -------------- //

    /// Default constructor.
#ifdef __cpp_concepts
    Covariance() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    Covariance()
#endif
      : Base {} {}


    /**
     * \brief Construct from another non-square-root \ref OpenKalman::covariance "covariance".
     */
#ifdef __cpp_concepts
    template<self_adjoint_covariance M> requires (not std::derived_from<std::decay_t<M>, Covariance>) and
      requires(M&& m) { Base {std::forward<M>(m)}; }
#else
    template<typename M, std::enable_if_t<self_adjoint_covariance<M> and
      (not std::is_base_of_v<Covariance, std::decay_t<M>>) and std::is_constructible_v<Base, M&&>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref OpenKalman::covariance_nestable "covariance_nestable".
     */
#ifdef __cpp_concepts
    template<covariance_nestable M> requires requires(M&& m) { Base {std::forward<M>(m)}; }
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and std::is_constructible_v<Base, M&&>, int> = 0>
#endif
    explicit Covariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref OpenKalman::typed_matrix "typed_matrix".
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be self-adjoint, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix M> requires (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<row_coefficient_types_of_t<M>, TypedIndex> and
      requires(M&& m) { Base {oin::to_covariance_nestable<NestedSelfAdjoint>(std::forward<M>(m))}; }
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<row_coefficient_types_of_t<M>, TypedIndex> and
      std::is_constructible_v<Base,
        decltype(oin::to_covariance_nestable<NestedSelfAdjoint>(std::declval<M&&>()))>, int> = 0>
#endif
    explicit Covariance(M&& m) noexcept
      : Base {oin::to_covariance_nestable<NestedSelfAdjoint>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be self-adjoint, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      requires(M&& m) { Base {oin::to_covariance_nestable<NestedSelfAdjoint>(std::forward<M>(m))};
      }
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      std::is_constructible_v<Base,
        decltype(oin::to_covariance_nestable<NestedSelfAdjoint>(std::declval<M&&>()))>, int> = 0>
#endif
    explicit Covariance(M&& m) noexcept
      : Base {oin::to_covariance_nestable<NestedSelfAdjoint>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a row-major list of Scalar coefficients forming a self-adjoint matrix.
     * \details The number of coefficients must match the size of the matrix (or the number of rows, if
     * NestedMatrix is diagonal).
     * The matrix is assumed (without enforcement) to be self-adjoint, but only the data in the lower-left triangle
     * is significant.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { Base {MatrixTraits<std::decay_t<NestedSelfAdjoint>>::make(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, const Scalar> and ...) and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == dim) or
        (sizeof...(Args) == dim * dim)) and std::is_constructible_v<Base, NestedSelfAdjoint&&>, int> = 0>
#endif
    Covariance(Args ... args)
      : Base {MatrixTraits<std::decay_t<NestedSelfAdjoint>>::make(static_cast<const Scalar>(args)...)} {}


    // ---------------------- //
    //  Assignment Operators  //
    // ---------------------- //

    /// Assign from a compatible \ref OpenKalman::covariance "covariance".
#ifdef __cpp_concepts
    template<self_adjoint_covariance Arg>
    requires (not std::derived_from<std::decay_t<Arg>, Covariance>) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<Covariance, std::decay_t<Arg>>) and
      (self_adjoint_covariance<Arg> and equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, nested_matrix_of_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(arg));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible square \ref OpenKalman::typed_matrix "typed_matrix".
     * \note This assumes, without checking, that it is self-adjoint.
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires square_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, NestedSelfAdjoint>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg> and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex> and
      modifiable<NestedMatrix, NestedSelfAdjoint>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(oin::to_covariance_nestable<NestedSelfAdjoint>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /// Assign from a compatible \ref OpenKalman::covariance_nestable "covariance_nestable".
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<(covariance_nestable<Arg>) and modifiable<NestedMatrix, Arg>, int> = 0>
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
     * \brief Assign from a compatible square \ref OpenKalman::typed_matrix_nestable "typed_matrix_nestable".
     * \note This assumes, without checking, that it is self-adjoint.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and square_matrix<Arg> and
      modifiable<NestedMatrix, NestedSelfAdjoint>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
      square_matrix<Arg> and modifiable<NestedMatrix, NestedSelfAdjoint>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(oin::to_covariance_nestable<NestedSelfAdjoint>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Increment by another non-square-root \ref OpenKalman::covariance "covariance" or
     * square \ref OpenKalman::typed_matrix "typed_matrix".
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      (self_adjoint_covariance<Arg> or (typed_matrix<Arg> and square_matrix<Arg>)) and
        equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      (self_adjoint_covariance<Arg> or (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>, int> = 0>
#endif
    auto& operator+=(Arg&& arg)
    {
      if constexpr(hermitian_matrix<NestedMatrix>)
      {
        nested_matrix() += oin::to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg));
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0)
        {
          if constexpr(triangular_matrix<NestedMatrix, TriangleType::upper>)
          {

            nested_matrix() = QR_decomposition(concatenate_vertical(
              nested_matrix(), oin::to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))));
          }
          else
          {
            nested_matrix() = LQ_decomposition(concatenate_horizontal(
              nested_matrix(), oin::to_covariance_nestable<NestedMatrix>(std::forward<Arg>(arg))));
          }
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() += oin::to_covariance_nestable<NestedSelfAdjoint>(
            std::forward<Arg>(arg));
        }
      }
      return *this;
    }


    /**
     * Increment by another Covariance of the same type.
     */
#ifdef __cpp_concepts
    auto& operator+=(const Covariance& arg) requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>), int> = 0>
    auto& operator+=(const Covariance& arg)
#endif
    {
      return operator+=<const Covariance&>(arg);
    }


#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      (self_adjoint_covariance<Arg> or (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      (self_adjoint_covariance<Arg> or (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<row_coefficient_types_of_t<Arg>, TypedIndex>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      if constexpr(hermitian_matrix<NestedMatrix>)
      {
        nested_matrix() -= oin::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0)
        {
          using TLowerType = typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<TriangleType::lower>;
          const auto U = oin::to_covariance_nestable<TLowerType>(arg);
          OpenKalman::rank_update(nested_matrix(), U, Scalar(-1));
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() -= oin::to_covariance_nestable<NestedSelfAdjoint>(arg);
        }
      }
      return *this;
    }


    /**
     * Decrement by another Covariance of the same type.
     */
#ifdef __cpp_concepts
    auto& operator-=(const Covariance& arg) requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>), int> = 0>
    auto& operator-=(const Covariance& arg)
#endif
    {
      return operator-=<const Covariance&>(arg);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(hermitian_matrix<NestedMatrix>)
      {
        nested_matrix() *= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else if (s > 0)
      {
        if (synchronization_direction() >= 0) nested_matrix() *= square_root(static_cast<const Scalar>(s));
        if (synchronization_direction() <= 0) cholesky_nested_matrix() *= static_cast<const Scalar>(s);
      }
      else if (s < 0)
      {
        if (synchronization_direction() >= 0)
        {
          using TLowerType = typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<TriangleType::lower>;
          const auto U = oin::to_covariance_nestable<TLowerType>(*this);
          nested_matrix() = make_zero_matrix_like(nested_matrix());
          OpenKalman::rank_update(nested_matrix(), U, s);
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() *= static_cast<const Scalar>(s);
        }
      }
      else
      {
        nested_matrix() = make_zero_matrix_like(nested_matrix());
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() = make_zero_matrix_like(cholesky_nested_matrix());
          mark_synchronized();
        }
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
      if constexpr(hermitian_matrix<NestedMatrix>)
      {
        nested_matrix() /= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else if (s > 0)
      {
        if (synchronization_direction() >= 0) nested_matrix() /= square_root(static_cast<const Scalar>(s));
        if (synchronization_direction() <= 0) cholesky_nested_matrix() /= static_cast<const Scalar>(s);
      }
      else if (s < 0)
      {
        if (synchronization_direction() >= 0)
        {
          using TLowerType = typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularMatrixFrom<TriangleType::lower>;
          const auto u = oin::to_covariance_nestable<TLowerType>(*this);
          nested_matrix() = make_zero_matrix_like(nested_matrix());
          OpenKalman::rank_update(nested_matrix(), u, 1 / static_cast<const Scalar>(s));
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() /= static_cast<const Scalar>(s);
        }
      }
      else
      {
        throw (std::runtime_error("Covariance operator/=: divide by zero"));
      }
      return *this;
    }


    /// Scale by a factor. Equivalent to multiplication by the square of a scalar.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& scale(const S s)
    {
      if constexpr(hermitian_matrix<NestedMatrix>)
      {
        nested_matrix() *= static_cast<const Scalar>(s) * s;
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0) nested_matrix() *= static_cast<const Scalar>(s);
        if (synchronization_direction() <= 0) cholesky_nested_matrix() *= static_cast<const Scalar>(s) * s;
      }
      return *this;
    }


    /// Scale by the inverse of a scalar factor. Equivalent by division by the square of a scalar.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& inverse_scale(const S s)
    {
      if constexpr(hermitian_matrix<NestedMatrix>)
      {
        nested_matrix() /= static_cast<const Scalar>(s) * s;
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0) nested_matrix() /= static_cast<const Scalar>(s);
        if (synchronization_direction() <= 0) cholesky_nested_matrix() /= static_cast<const Scalar>(s) * s;
      }
      return *this;
    }


    // ------- //
    //  Other  //
    // ------- //

    /**
     * \brief Take the Cholesky square root of *this.
     * \details If *this is an lvalue reference, this creates a reference to the nested matrix rather than a copy.
     * \return A SquareRootCovariance based on *this.
     * \note One cannot assume that the lifetime of the result is longer than the lifetime of the object.
     */
    auto square_root() &
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return SquareRootCovariance<TypedIndex, std::remove_reference_t<NestedMatrix>&> {*this};
      }
      else
      {
        static_assert(diagonal_matrix<NestedMatrix>);
        auto n = Cholesky_factor<storage_triangle>(nested_matrix());
        return SquareRootCovariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square_root() const &
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return SquareRootCovariance<TypedIndex, const std::remove_reference_t<NestedMatrix>&> {*this};
      }
      else
      {
        static_assert(diagonal_matrix<NestedMatrix>);
        auto n = Cholesky_factor<storage_triangle>(nested_matrix());
        return SquareRootCovariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square_root() &&
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return SquareRootCovariance<TypedIndex, std::remove_reference_t<NestedMatrix>> {std::move(*this)};
      }
      else
      {
        static_assert(diagonal_matrix<NestedMatrix>);
        auto n = Cholesky_factor<storage_triangle>(std::move(*this).nested_matrix());
        return SquareRootCovariance<TypedIndex, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square_root() const &&
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
      {
        return SquareRootCovariance<TypedIndex, std::remove_reference_t<NestedMatrix>> {std::move(*this)};
      }
      else
      {
        static_assert(diagonal_matrix<NestedMatrix>);
        auto n = Cholesky_factor<storage_triangle>(std::move(*this).nested_matrix());
        return SquareRootCovariance<TypedIndex, decltype(n)> {std::move(n)};
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
      if (synchronization_direction() < 0) Base::synchronize_reverse();
      if constexpr (one_by_one_matrix<NestedMatrix>)
      {
        Base::operator()(0, 0) = trace(nested_matrix()) + alpha * trace(u * adjoint(u));
      }
      else
      {
        OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha);
      }
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
      if constexpr (one_by_one_matrix<NestedMatrix>)
      {
        auto b = trace(nested_matrix()) + alpha * trace(u * adjoint(u));
        return make(MatrixTraits<std::decay_t<NestedMatrix>>::make(b));
      }
      else
      {
        return make(OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha));
      }
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
    friend struct Covariance;


#ifdef __cpp_concepts
    template<fixed_index_descriptor C, covariance_nestable N> requires
      (dimension_size_of_v<C> == row_dimension_of_v<N>) and (not std::is_rvalue_reference_v<N>)
#else
    template<typename, typename>
#endif
    friend struct SquareRootCovariance;

  };


  // ------------------------------- //
  //        Deduction guides         //
  // ------------------------------- //

  /**
   * \brief Deduce Covariance type from a \ref covariance_nestable.
   */
#ifdef __cpp_concepts
  template<covariance_nestable M>
#else
  template<typename M, std::enable_if_t<covariance_nestable<M>, int> = 0>
#endif
  explicit Covariance(M&&) -> Covariance<Dimensions<row_dimension_of_v<M>>, passable_t<M>>;


  /**
   * \brief Deduce Covariance type from a square \ref typed_matrix
   */
#ifdef __cpp_concepts
  template<typed_matrix M> requires square_matrix<M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M> and square_matrix<M>, int> = 0>
#endif
  explicit Covariance(M&&) -> Covariance<
    row_coefficient_types_of_t<M>,
    typename MatrixTraits<std::decay_t<nested_matrix_of_t<M>>>::template SelfAdjointMatrixFrom<>>;


  /**
   * \brief Deduce Covariance type from a square \ref typed_matrix_nestable
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and square_matrix<M>
#else
  template<typename M, std::enable_if_t<
    typed_matrix_nestable<M> and (not covariance_nestable<M>) and square_matrix<M>, int> = 0>
#endif
  explicit Covariance(M&&) -> Covariance<
    Dimensions<row_dimension_of_v<M>>, typename MatrixTraits<std::decay_t<M>>::template SelfAdjointMatrixFrom<>>;


  // ---------------- //
  //  Make Functions  //
  // ---------------- //

  /**
   * \brief Make a Covariance from a \ref covariance_nestable, specifying the fixed_index_descriptor.
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
  make_covariance(Arg&& arg) noexcept
  {
    return Covariance<TypedIndex, passable_t<Arg>> {std::forward<Arg>(arg)};
  }


  /**
   * \brief Make a Covariance from a \ref covariance_nestable, specifying the fixed_index_descriptor.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower, diagonal).
   * \tparam Arg A \ref covariance_nestable with size matching TypedIndex.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, TriangleType triangle_type, covariance_nestable Arg> requires
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<Arg>) and
    (triangle_type != TriangleType::lower or triangular_matrix<Arg, TriangleType::lower>) and
    (triangle_type != TriangleType::upper or triangular_matrix<Arg, TriangleType::upper>) and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<Arg>)
#else
  template<typename TypedIndex, TriangleType triangle_type, typename Arg, std::enable_if_t<
    fixed_index_descriptor<TypedIndex> and covariance_nestable<Arg> and
    (dimension_size_of_v<TypedIndex> == row_dimension_of<Arg>::value) and
    (triangle_type != TriangleType::lower or triangular_matrix<Arg, TriangleType::lower>) and
    (triangle_type != TriangleType::upper or triangular_matrix<Arg, TriangleType::upper>) and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<Arg>), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    return Covariance<TypedIndex, passable_t<Arg>> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   * \brief Make a Covariance from a \ref covariance_nestable, with default Axis coefficients.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable.
   */
#ifdef __cpp_concepts
  template<covariance_nestable Arg>
#else
  template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a Covariance (with nested triangular matrix) from a self-adjoint \ref typed_matrix_nestable.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable with size matching TypedIndex.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, TriangleType triangle_type, typed_matrix_nestable Arg> requires
    (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<Arg>) and (dimension_size_of_v<TypedIndex> == column_dimension_of_v<Arg>)
#else
  template<typename TypedIndex, TriangleType triangle_type, typename Arg, std::enable_if_t<
    fixed_index_descriptor<TypedIndex> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (dimension_size_of_v<TypedIndex> == row_dimension_of<Arg>::value) and
    (dimension_size_of_v<TypedIndex> == column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using T = typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<triangle_type>;
    return Covariance<TypedIndex, T> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   * \brief Make a Covariance from a self-adjoint \ref typed_matrix_nestable, specifying the coefficients.
   * \tparam TypedIndex The coefficient types corresponding to the rows and columns.
   * \tparam Arg A square \ref typed_matrix_nestable with size matching TypedIndex.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<Arg>) and (dimension_size_of_v<TypedIndex> == column_dimension_of_v<Arg>)
#else
  template<typename TypedIndex, typename Arg, std::enable_if_t<
    fixed_index_descriptor<TypedIndex> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (dimension_size_of_v<TypedIndex> == row_dimension_of<Arg>::value) and
    (dimension_size_of_v<TypedIndex> == column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using SA = typename MatrixTraits<std::decay_t<Arg>>::template SelfAdjointMatrixFrom<>;
    return make_covariance<TypedIndex, SA>(oin::to_covariance_nestable<SA>(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a default Axis Covariance (with nested triangular matrix) from a self-adjoint \ref typed_matrix_nestable.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
    (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_covariance<C, triangle_type>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a Covariance from a self-adjoint \ref typed_matrix_nestable, using default Axis coefficients.
   * \tparam Arg A square \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance from a \ref covariance_nestable or \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typename Arg> requires
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename TypedIndex, typename Arg, std::enable_if_t<
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    constexpr TriangleType triangle_type = triangle_type_of_v<typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<>>;
    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<std::decay_t<Arg>>::template DiagonalMatrixFrom<>,
      std::conditional_t<triangular_matrix<Arg>,
        typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<triangle_type>,
        typename MatrixTraits<std::decay_t<Arg>>::template SelfAdjointMatrixFrom<triangle_type>>>;
    return Covariance<TypedIndex, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance with a nested triangular matrix, from a \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, TriangleType triangle_type, typed_matrix_nestable Arg> requires
    square_matrix<Arg>
#else
  template<typename TypedIndex, TriangleType triangle_type, typename Arg,
    std::enable_if_t<fixed_index_descriptor<TypedIndex> and typed_matrix_nestable<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using B = std::conditional_t<triangle_type == TriangleType::diagonal,
      typename MatrixTraits<std::decay_t<Arg>>::template DiagonalMatrixFrom<>,
      typename MatrixTraits<std::decay_t<Arg>>::template TriangularMatrixFrom<triangle_type>>;
    return Covariance<TypedIndex, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance from a \ref typed_matrix_nestable or \ref covariance_nestable.
   * \details The coefficients will be Axis.
   */
#ifdef __cpp_concepts
  template<typename Arg> requires (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<(covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_covariance<C, Arg>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance based on a nested triangle, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix_nestable Arg> requires square_matrix<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and
    square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using C = Dimensions<row_dimension_of_v<Arg>>;
    return make_covariance<C, triangle_type, Arg>();
  }


  /**
   * \overload
   * \brief Make a Covariance based on another non-square-root \ref covariance.
   */
#ifdef __cpp_concepts
  template<self_adjoint_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<self_adjoint_covariance<Arg>, int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = row_coefficient_types_of_t<Arg>;
    return Covariance<C, nested_matrix_of_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance from a non-square-root \ref covariance.
   */
#ifdef __cpp_concepts
  template<self_adjoint_covariance T>
#else
  template<typename T, std::enable_if_t<self_adjoint_covariance<T>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using C = row_coefficient_types_of_t<T>;
    using B = nested_matrix_of_t<T>;
    return make_covariance<C, B>();
  }


  /**
   * \overload
   * \brief Make a Covariance from a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = row_coefficient_types_of_t<Arg>;
    return make_covariance<C>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a Covariance, with a nested triangular matrix, from a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg> requires
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = row_coefficient_types_of_t<Arg>;
    return make_covariance<C, triangle_type>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance, with nested triangular type based on a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg> requires square_matrix<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using C = row_coefficient_types_of_t<Arg>;
    using B = nested_matrix_of_t<Arg>;
    return make_covariance<C, triangle_type, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance, based on a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using C = row_coefficient_types_of_t<Arg>;
    using B = nested_matrix_of_t<Arg>;
    return make_covariance<C, B>();
  }


} // OpenKalman

#endif //OPENKALMAN_COVARIANCE_HPP
