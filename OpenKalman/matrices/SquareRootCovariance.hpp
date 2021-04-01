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

#include <mutex>

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix> requires
    (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct SquareRootCovariance : internal::CovarianceBase<SquareRootCovariance<Coefficients, NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<Coefficients>);
    static_assert(covariance_nestable<NestedMatrix>);
    static_assert(Coefficients::size == MatrixTraits<NestedMatrix>::dimension);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    // May be accessed externally through MatrixTraits:
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this matrix.

  private:

    using Base = internal::CovarianceBase<SquareRootCovariance, NestedMatrix>;

    // May be accessed externally through MatrixTraits:
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;

    // May be accessed externally through MatrixTraits:
    static constexpr TriangleType triangle_type =
      triangle_type_of<typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>;

    // A triangular nested matrix type.
    using NestedTriangular = std::conditional_t<diagonal_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<triangle_type>>;


    // A function that makes a self-contained covariance from a nested matrix.
    template<typename C = Coefficients, typename Arg>
    static constexpr auto make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


    using typename Base::CholeskyNestedMatrix;
    using Base::nested_matrix;
    using Base::cholesky_nested_matrix;
    using Base::synchronization_direction;
    using Base::synchronize_forward;
    using Base::synchronize_reverse;
    using Base::mark_nested_matrix_changed;
    using Base::mark_cholesky_nested_matrix_changed;
    using Base::mark_synchronized;

    // Mutex for all updates to the nested matrices.
    mutable std::mutex nested_mutex;


    /**
     * \brief Construct from a non-square-root, non-diagonal \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance M> requires (not square_root_covariance<M>) and
      (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (self_adjoint_matrix<nested_matrix_t<M>> == self_adjoint_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<covariance<M> and (not square_root_covariance<M>) and
      (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (self_adjoint_matrix<nested_matrix_t<M>> == self_adjoint_matrix<NestedMatrix>), int> = 0>
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


    /// Non-const copy constructor.
    SquareRootCovariance(SquareRootCovariance& other) : Base {other} {}


    /// Const copy constructor.
    SquareRootCovariance(const SquareRootCovariance& other) : Base {other} {}


    /// Move constructor.
    SquareRootCovariance(SquareRootCovariance&& other) : Base {std::move(other)} {}


    /**
     * \brief Construct from another \ref square_root_covariance.
     */
#ifdef __cpp_concepts
    template<square_root_covariance M> requires (not std::derived_from<std::decay_t<M>, SquareRootCovariance>) and
      internal::same_triangle_type_as<M, SquareRootCovariance> and std::is_constructible_v<Base, M>
#else
    template<typename M, std::enable_if_t<
      square_root_covariance<M> and (not std::is_base_of_v<SquareRootCovariance, std::decay_t<M>>) and
      internal::same_triangle_type_as<M, SquareRootCovariance> and std::is_constructible_v<Base, M>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable M> requires std::is_constructible_v<Base, M>
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and std::is_constructible_v<Base, M>, int> = 0>
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
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedTriangular>(std::declval<M>()))>
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedTriangular>(std::declval<M>()))>,
        int> = 0>
#endif
    explicit SquareRootCovariance(M&& m) noexcept
      : Base {internal::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a \ref typed_matrix_nestable.
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be triangular, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedTriangular>(std::declval<M>()))>
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedTriangular>(std::declval<M>()))>,
        int> = 0>
#endif
    explicit SquareRootCovariance(M&& m) noexcept
      : Base {internal::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))} {}


    /// Construct from Scalar coefficients. Assumes matrix is triangular, and only reads lower left triangle.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (sizeof...(Args) != dimension or diagonal_matrix<NestedMatrix>) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<NestedTriangular>::make(static_cast<const Scalar>(std::declval<const Args>())...))>; }
    SquareRootCovariance(const Args ... args)
      : Base {MatrixTraits<NestedTriangular>::make(static_cast<const Scalar>(args)...)} {}
#else
    // Note: std::is_constructible_v cannot be used here with ::make.
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, Scalar> and ...) and
      (sizeof...(Args) == dimension) and diagonal_matrix<NestedMatrix> and
      std::is_constructible_v<Base, NestedTriangular&&>, int> = 0>
    SquareRootCovariance(const Args ... args)
      : Base {MatrixTraits<NestedTriangular>::make(static_cast<const Scalar>(args)...)} {}

    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, Scalar> and ...) and
      (sizeof...(Args) == dimension * dimension) and std::is_constructible_v<Base, NestedTriangular&&>, int> = 0>
    SquareRootCovariance(const Args ... args)
      : Base {MatrixTraits<NestedTriangular>::make(static_cast<const Scalar>(args)...)} {}
#endif


    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const SquareRootCovariance& other)
#ifdef __cpp_concepts
    requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#endif
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>,
        "Assignment is not allowed because NestedMatrix is const.");
      std::scoped_lock lock {nested_mutex};
      if constexpr(not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(other);
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(SquareRootCovariance&& other) noexcept
#ifdef __cpp_concepts
    requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#endif
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>,
        "Assignment is not allowed because NestedMatrix is const.");
      std::scoped_lock lock {nested_mutex};
      if constexpr(not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(std::move(other));
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref square_root_covariance.
     * \note the triangle types must match.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires (not std::derived_from<std::decay_t<Arg>, SquareRootCovariance>) and
      internal::same_triangle_type_as<Arg, SquareRootCovariance> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      (not std::is_base_of_v<SquareRootCovariance, std::decay_t<Arg>>) and
      internal::same_triangle_type_as<Arg, SquareRootCovariance> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        std::scoped_lock lock {nested_mutex};
        Base::operator=(std::forward<Arg>(other));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref typed_matrix (assumed, without checking, to be triangular).
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires square_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, NestedTriangular>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, NestedTriangular>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        std::scoped_lock lock {nested_mutex};
        Base::operator=(internal::to_covariance_nestable<NestedTriangular>(std::forward<Arg>(other)));
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
        std::scoped_lock lock {nested_mutex};
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
        std::scoped_lock lock {nested_mutex};
        Base::operator=(internal::to_covariance_nestable<NestedTriangular>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * Increment by another \ref square_root_covariance or triangular \ref typed_matrix.
     * \warning This is computationally expensive if the nested matrix is not \ref triangular_matrix.
     * This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((square_root_covariance<Arg> and internal::same_triangle_type_as<Arg, SquareRootCovariance>) or
        (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((square_root_covariance<Arg> and internal::same_triangle_type_as<Arg, SquareRootCovariance>) or
        (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr(triangular_matrix<NestedMatrix>) // Case 1 or 2
      {
        nested_matrix() += internal::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else // Case 3 or 4
      {
        if (synchronization_direction() > 0) synchronize_forward();
        cholesky_nested_matrix() += internal::to_covariance_nestable<NestedTriangular>(arg);
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
     * Decrement by another \ref square_root_covariance or triangular \ref typed_matrix.
     * \warning This is computationally expensive if the nested matrix is not \ref triangular_matrix.
     * This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
    ((square_root_covariance<Arg> and internal::same_triangle_type_as<Arg, SquareRootCovariance>) or
      (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((square_root_covariance<Arg> and internal::same_triangle_type_as<Arg, SquareRootCovariance>) or
        (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() -= internal::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        cholesky_nested_matrix() -= internal::to_covariance_nestable<NestedTriangular>(arg);
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
      std::scoped_lock lock {nested_mutex};
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
      std::scoped_lock lock {nested_mutex};
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
     * \brief Multiply by another \ref square_root_covariance.
     * \details The underlying triangle type (upper or lower) of Arg much match that of the nested matrix.
     * \warning This is computationally expensive unless the nested matrices of *this and Arg are both either
     * triangular or self-adjoint.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires internal::same_triangle_type_as<SquareRootCovariance, Arg> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator*=(const Arg& arg)
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() *= internal::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        cholesky_nested_matrix() *= internal::to_covariance_nestable<NestedTriangular>(arg);
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


    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }


    /**
     * \brief Take the Cholesky square of *this.
     * \details If *this is an lvalue reference, this creates a reference to the nested matrix rather than a copy.
     * \return A Covariance based on *this.
     */
#ifdef __cpp_concepts
    auto square() & requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square() &
#endif
    {
      return Covariance<Coefficients, std::add_lvalue_reference_t<NestedMatrix>>(*this);
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square() && requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square() &&
#endif
    {
      return Covariance<Coefficients, self_contained_t<NestedMatrix>>(std::move(*this));
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square() const & requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square() const &
#endif
    {
      return Covariance<Coefficients, std::add_lvalue_reference_t<const NestedMatrix>>(*this);
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square() const && requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square() const &&
#endif
    {
      return Covariance<Coefficients, const self_contained_t<NestedMatrix>>(std::move(*this));
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square() const & requires diagonal_matrix<NestedMatrix> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<diagonal_matrix<T> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>), int> = 0>
    auto square() const &
#endif
    {
      auto n = make_self_contained(Cholesky_square(nested_matrix()));
      return Covariance<Coefficients, decltype(n)> {std::move(n)};
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square() const && requires diagonal_matrix<NestedMatrix> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<diagonal_matrix<T> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>), int> = 0>
    auto square() const &&
#endif
    {
      constexpr decltype(auto) fw = [] (auto&& n) { return std::forward<decltype(n)>(n); };
      auto n = make_self_contained(Cholesky_square(fw(nested_matrix())));
      return Covariance<Coefficients, decltype(n)> {std::move(n)};
    }


    /**
     * \brief Make a Covariance based on an operation on the nested matrices.
     * \tparam F1 Operation on NestedMatrix.
     * \tparam F2 Operation on the return value of cholesky_nested_matrix
     */
#ifdef __cpp_concepts
    template<typename F1, typename F2> requires
      std::invocable<F1, const NestedMatrix&> and std::invocable<F2, const NestedTriangular&>
#else
    template<typename F1, typename F2, std::enable_if_t<
      std::is_invocable_v<F1, const NestedMatrix&> and std::is_invocable_v<F2, const NestedTriangular&>, int> = 0>
#endif
    auto covariance_op(F1&& f1, F2&& f2) const
    {
      auto n = make_self_contained(f1(nested_matrix()));
      using N = decltype(n);
      if constexpr (internal::case1or2<SquareRootCovariance, N>)
      {
        return make(std::move(n));
      }
      else
      {
        std::scoped_lock lock {nested_mutex};
        if (synchronization_direction() >= 0)
        {
          auto r = make(std::move(n));
          if (synchronization_direction() == 0)
          {
            r.mark_synchronized();
            if (r.synchronization_direction() <= 0) r.cholesky_nested_matrix() = f2(cholesky_nested_matrix());
          }
          return r;
        }
        else
        {
          return SquareRootCovariance<Coefficients, N> {
            internal::to_covariance_nestable<N>(f2(cholesky_nested_matrix()))};
        }
      }
    }


    /**
     * \brief Perform a rank update.
     */
#ifdef __cpp_concepts
    template<typed_matrix U> requires equivalent_to<typename MatrixTraits<U>::RowCoefficients, Coefficients> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      equivalent_to<typename MatrixTraits<U>::RowCoefficients, Coefficients> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& rank_update(const U& u, const Scalar alpha = 1) &
    {
      std::scoped_lock lock {nested_mutex};
      if (synchronization_direction() < 0) synchronize_reverse();
      OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha);
      mark_nested_matrix_changed();
      return *this;
    }


    /**
     * \brief Perform a rank update.
     */
#ifdef __cpp_concepts
    template<typed_matrix U> requires equivalent_to<typename MatrixTraits<U>::RowCoefficients, Coefficients>
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      equivalent_to<typename MatrixTraits<U>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto rank_update(const U& u, const Scalar alpha = 1) const
    {
      std::scoped_lock lock {nested_mutex};
      if (synchronization_direction() < 0) synchronize_reverse();
      return make(OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha));
    }


    /**
     * \return The nested matrix, potentially converted to triangular form.
     */
    decltype(auto) get_triangular_nested_matrix() const &
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr (triangular_matrix<NestedMatrix>)
      {
        if (synchronization_direction() < 0) synchronize_reverse();
        return nested_matrix();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        return cholesky_nested_matrix();
      }
    }

    /// \overload
    decltype(auto) get_triangular_nested_matrix() const &&
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr (triangular_matrix<NestedMatrix>)
      {
        if (synchronization_direction() < 0) synchronize_reverse();
        if constexpr (std::is_lvalue_reference_v<NestedMatrix>) return nested_matrix();
        else return std::move(nested_matrix());
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        auto ret = cholesky_nested_matrix();
        return ret;
      }
    }


    /**
     * \return The nested matrix, potentially converted to self-adjoint form.
     */
    decltype(auto) get_self_adjoint_nested_matrix() const &
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr (self_adjoint_matrix<NestedMatrix>)
      {
        if (synchronization_direction() < 0) synchronize_reverse();
        return nested_matrix();
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        return cholesky_nested_matrix();
      }
    }

    /// \overload
    decltype(auto) get_self_adjoint_nested_matrix() const &&
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr (self_adjoint_matrix<NestedMatrix>)
      {
        if (synchronization_direction() < 0) synchronize_reverse();
        if constexpr (std::is_lvalue_reference_v<NestedMatrix>) return nested_matrix();
        else return std::move(nested_matrix());
      }
      else
      {
        if (synchronization_direction() > 0) synchronize_forward();
        auto ret = cholesky_nested_matrix();
        return ret;
      }
    }


    /**
     * \brief Get or set element (i, j) of the covariance matrix.
     * \param i The row.
     * \param j The column.
     * \return An ElementSetter object.
     */
#ifdef __cpp_concepts
    auto operator() (std::size_t i, std::size_t j)
    requires element_gettable<NestedMatrix, 2>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 2>, int> = 0>
    auto operator() (std::size_t i, std::size_t j)
#endif
    {
      return Base::operator()(i, j);
    }

    /// \overload
#ifdef __cpp_concepts
    auto operator() (std::size_t i, std::size_t j) const
    requires element_gettable<NestedMatrix, 2>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 2>, int> = 0>
    auto operator() (std::size_t i, std::size_t j) const
#endif
    {
      return Base::operator()(i, j);
    }


    /**
     * \brief Get or set element i of the covariance matrix, if it is a vector.
     * \param i The row.
     * \return An ElementSetter object.
     */
#ifdef __cpp_concepts
    auto operator[] (std::size_t i)
    requires element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 1>, int> = 0>
    auto operator[] (std::size_t i)
#endif
    {
      return Base::operator[](i);
    }


    /// \overload
#ifdef __cpp_concepts
    auto operator[] (std::size_t i) const
    requires element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 1>, int> = 0>
    auto operator[] (std::size_t i) const
#endif
    {
      return Base::operator[](i);
    }

    /// \overload
#ifdef __cpp_concepts
    auto operator() (std::size_t i)
    requires element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 1>, int> = 0>
    auto operator() (std::size_t i)
#endif
    {
      return operator[](i);
    }

    /// \overload
#ifdef __cpp_concepts
    auto operator() (std::size_t i) const
    requires element_gettable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_gettable<T, 1>, int> = 0>
    auto operator() (std::size_t i) const
#endif
    {
      return operator[](i);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
#ifdef __cpp_concepts
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
    requires element_settable<NestedMatrix, 2>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 2>, int> = 0>
    void set_element(const Scalar s, const std::size_t i, const std::size_t j)
#endif
    {
      std::scoped_lock lock {nested_mutex};
      Base::set_element(s, i, j);
    }


    /**
     * \brief Set an element of the cholesky nested matrix.
     */
#ifdef __cpp_concepts
    void set_element(const Scalar s, const std::size_t i)
    requires element_settable<NestedMatrix, 1>
#else
    template<typename T = NestedMatrix, std::enable_if_t<element_settable<T, 1>, int> = 0>
    void set_element(const Scalar s, const std::size_t i)
#endif
    {
      std::scoped_lock lock {nested_mutex};
      Base::set_element(s, i);
    }


  private:

#ifdef __cpp_concepts
    template<typename, typename>
    friend struct internal::CovarianceBase;
#else
    template<typename, typename, typename>
    friend struct internal::CovarianceBase;
#endif


#ifdef __cpp_concepts
    template<coefficients C, covariance_nestable N> requires
    (C::size == MatrixTraits<N>::dimension) and (not std::is_rvalue_reference_v<N>)
#else
    template<typename, typename>
#endif
    friend struct SquareRootCovariance;


#ifdef __cpp_concepts
    template<coefficients C, covariance_nestable N> requires
      (C::size == MatrixTraits<N>::dimension) and (not std::is_rvalue_reference_v<N>)
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
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<Axes<MatrixTraits<M>::dimension>, passable_t<M>>;


  /**
   * \brief Deduce SquareRootCovariance type from a square \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix M> requires square_matrix<M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M> and square_matrix<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<nested_matrix_t<M>>::template TriangularMatrixFrom<>>;


  /**
   * \brief Deduce SquareRootCovariance type from a square \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and square_matrix<M>
#else
  template<typename M, std::enable_if_t<
    typed_matrix_nestable<M> and (not covariance_nestable<M>) and square_matrix<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<Axes<MatrixTraits<M>::dimension>,
    typename MatrixTraits<M>::template TriangularMatrixFrom<>>;


  // ---------------- //
  //  Make Functions  //
  // ---------------- //

  /**
   * \brief Make a SquareRootCovariance from a \ref covariance_nestable, specifying the coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable with size matching Coefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable Arg> requires
    (Coefficients::size == MatrixTraits<Arg>::dimension)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    covariance_nestable<Arg> and (Coefficients::size == MatrixTraits<Arg>::dimension), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    return SquareRootCovariance<Coefficients, passable_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance from a \ref covariance_nestable, with default Axis coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
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
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_square_root_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance (with nested triangular matrix) from a self-adjoint \ref typed_matrix_nestable.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable with size matching Coefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type = TriangleType::lower, typed_matrix_nestable Arg>
  requires (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (Coefficients::size == MatrixTraits<Arg>::dimension) and (Coefficients::size == MatrixTraits<Arg>::columns)
#else
  template<typename Coefficients, TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (Coefficients::size == MatrixTraits<Arg>::dimension) and
    (Coefficients::size == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using T = typename MatrixTraits<Arg>::template TriangularMatrixFrom<triangle_type>;
    return SquareRootCovariance<Coefficients, T>(std::forward<Arg>(arg));
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
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_square_root_covariance<C, triangle_type>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, typed_matrix_nestable Arg> requires square_matrix<Arg>
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<coefficients<Coefficients> and typed_matrix_nestable<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using B = std::conditional_t<triangle_type == TriangleType::diagonal,
    typename MatrixTraits<Arg>::template DiagonalMatrixFrom<>,
      typename MatrixTraits<Arg>::template TriangularMatrixFrom<triangle_type>>;
    return SquareRootCovariance<Coefficients, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref covariance_nestable or \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename Arg> requires
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    constexpr TriangleType template_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularMatrixFrom<>>;
    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalMatrixFrom<>,
      std::conditional_t<self_adjoint_matrix<Arg>,
        typename MatrixTraits<Arg>::template SelfAdjointMatrixFrom<template_type>,
        typename MatrixTraits<Arg>::template TriangularMatrixFrom<template_type>>>;
    return SquareRootCovariance<Coefficients, B>();
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
    using C = Axes<MatrixTraits<Arg>::dimension>;
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
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_square_root_covariance<C, triangle_type, Arg>();
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance based on another \ref square_root_covariance.
   */
#ifdef __cpp_concepts
  template<square_root_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<square_root_covariance<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return SquareRootCovariance<C, nested_matrix_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref square_root_covariance.
   */
#ifdef __cpp_concepts
  template<square_root_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<square_root_covariance<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
    return make_square_root_covariance<C, B>();
  }


  ////////////////////////////
  //        Traits          //
  ////////////////////////////

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<SquareRootCovariance<Coeffs, ArgType>>
  {
    using NestedMatrix = ArgType;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr auto columns = dimension;
    static_assert(Coeffs::size == dimension);
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Coeffs;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this vector.

    static constexpr TriangleType
    triangle_type = triangle_type_of<typename MatrixTraits<ArgType>::template TriangularMatrixFrom<>>;

    template<std::size_t rows = dimension, std::size_t cols = rows, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, rows, cols, S>;

    using SelfContainedFrom = SquareRootCovariance<Coeffs, self_contained_t<NestedMatrix>>;


    /// Make SquareRootCovariance from a \ref covariance_nestable.
#ifdef __cpp_concepts
    template<coefficients C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


    static auto zero() { return SquareRootCovariance<Coeffs, std::decay_t<NestedMatrix>>::zero(); }

    static auto identity() { return SquareRootCovariance<Coeffs, std::decay_t<NestedMatrix>>::identity(); }
  };


}


#endif //OPENKALMAN_SQUAREROOTCOVARIANCE_HPP

