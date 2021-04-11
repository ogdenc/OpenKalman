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

#include <mutex>

namespace OpenKalman
{
  //////////////////
  //  Covariance  //
  //////////////////

#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix> requires
    (Coefficients::size == MatrixTraits<NestedMatrix>::rows) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct Covariance : internal::CovarianceBase<Covariance<Coefficients, NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(coefficients<Coefficients>);
    static_assert(covariance_nestable<NestedMatrix>);
    static_assert(Coefficients::size == MatrixTraits<NestedMatrix>::rows);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
#endif

    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this matrix.

  private:

    using Base = internal::CovarianceBase<Covariance, NestedMatrix>;

    // May be accessed externally through MatrixTraits:
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::rows;

    // May be accessed externally through MatrixTraits:
    static constexpr TriangleType storage_triangle =
      triangle_type_of<typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<>>;

    // A self-adjoint nested matrix type.
    using NestedSelfAdjoint = std::conditional_t<diagonal_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<storage_triangle>>;

    // A function that makes a covariance from a nested matrix.
    template<typename C = Coefficients, typename Arg>
    static auto make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
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
     * \brief Construct from a non-diagonal \ref square_root_covariance.
     */
#ifdef __cpp_concepts
    template<square_root_covariance M> requires (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (self_adjoint_matrix<nested_matrix_t<M>> == self_adjoint_matrix<NestedMatrix>) and
      std::is_constructible_v<Base, M>
#else
    template<typename M, std::enable_if_t<square_root_covariance<M> and
      (not diagonal_matrix<M> or identity_matrix<M> or zero_matrix<M>) and
      (self_adjoint_matrix<nested_matrix_t<M>> == self_adjoint_matrix<NestedMatrix>) and
      std::is_constructible_v<Base, M>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base {std::forward<M>(m)} {}

  public:

    // ------------ //
    // Constructors //
    // ------------ //

    /// Default constructor.
#ifdef __cpp_concepts
    Covariance() requires std::default_initializable<Base>
#else
    template<typename T = Base, std::enable_if_t<std::is_default_constructible_v<T>, int> = 0>
    Covariance()
#endif
      : Base() {}


    /// Non-const copy constructor.
    Covariance(Covariance& other) : Base(other) {}


    /// Const copy constructor.
    Covariance(const Covariance& other) : Base(other) {}


    /// Move constructor.
    Covariance(Covariance&& other) noexcept : Base(std::move(other)) {}


    /**
     * \brief Construct from another non-square-root \ref covariance.
     */
#ifdef __cpp_concepts
    template<covariance M> requires (not square_root_covariance<M>) and
      (not std::derived_from<std::decay_t<M>, Covariance>) and std::is_constructible_v<Base, M>
#else
    template<typename M, std::enable_if_t<covariance<M> and (not square_root_covariance<M>) and
      (not std::is_base_of_v<Covariance, std::decay_t<M>>) and std::is_constructible_v<Base, M>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable M> requires std::is_constructible_v<Base, M>
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and std::is_constructible_v<Base, M>, int> = 0>
#endif
    explicit Covariance(M&& m) noexcept : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref typed_matrix.
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be self-adjoint, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix M> requires (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedSelfAdjoint>(std::declval<M>()))>
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedSelfAdjoint>(std::declval<M>()))>,
        int> = 0>
#endif
    explicit Covariance(M&& m) noexcept
      : Base {internal::to_covariance_nestable<NestedSelfAdjoint>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a \ref typed_matrix_nestable.
     * \details M must be a \ref square_matrix, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be self-adjoint, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedSelfAdjoint>(std::declval<M>()))>
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and (not covariance_nestable<M>) and
      (square_matrix<M> or (diagonal_matrix<NestedMatrix> and column_vector<M>)) and
      std::is_constructible_v<Base, decltype(internal::to_covariance_nestable<NestedSelfAdjoint>(std::declval<M>()))>,
        int> = 0>
#endif
    explicit Covariance(M&& m) noexcept
      : Base {internal::to_covariance_nestable<NestedSelfAdjoint>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a row-major list of Scalar coefficients forming a self-adjoint matrix.
     * \details The number of coefficients must match the size of the matrix (or the number of rows, if
     * NestedMatrix is diagonal).
     * The matrix is assumed (without enforcement) to be self-adjoint, but only the data in the lower-left triangle
     * is significant.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> ... Args> requires (sizeof...(Args) > 0) and
      (sizeof...(Args) != dimension or diagonal_matrix<NestedMatrix>) and
      requires { std::is_constructible_v<Base,
        decltype(MatrixTraits<NestedSelfAdjoint>::make(static_cast<const Scalar>(std::declval<const Args>())...))>; }
    Covariance(const Args ... args)
      : Base {MatrixTraits<NestedSelfAdjoint>::make(static_cast<const Scalar>(args)...)} {}
#else
    // Note: std::is_constructible_v cannot be used here with ::make.
    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, Scalar> and ...) and
      (sizeof...(Args) == dimension) and diagonal_matrix<NestedMatrix> and
      std::is_constructible_v<Base, NestedSelfAdjoint&&>, int> = 0>
    Covariance(const Args ... args)
      : Base {MatrixTraits<NestedSelfAdjoint>::make(static_cast<const Scalar>(args)...)} {}

    template<typename ... Args, std::enable_if_t<(std::is_convertible_v<Args, Scalar> and ...) and
      (sizeof...(Args) == dimension * dimension) and (not one_by_one_matrix<NestedMatrix>) and
      std::is_constructible_v<Base, NestedSelfAdjoint&&>, int> = 0>
    Covariance(const Args ... args)
      : Base {MatrixTraits<NestedSelfAdjoint>::make(static_cast<const Scalar>(args)...)} {}
#endif


    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const Covariance& other)
#ifdef __cpp_concepts
      requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#endif
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>,
        "Assignment is not allowed because NestedMatrix is const.");
      std::scoped_lock lock {nested_mutex};
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) Base::operator=(other);
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(Covariance&& other) noexcept
#ifdef __cpp_concepts
      requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#endif
    {
      static_assert(not std::is_const_v<std::remove_reference_t<NestedMatrix>>,
        "Assignment is not allowed because NestedMatrix is const.");
      std::scoped_lock lock {nested_mutex};
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from a compatible \ref covariance.
#ifdef __cpp_concepts
    template<covariance Arg> requires (not square_root_covariance<Arg>) and
      (not std::derived_from<std::decay_t<Arg>, Covariance>) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, nested_matrix_t<Arg>>
#else
    template<typename Arg, std::enable_if_t<(not std::is_base_of_v<Covariance, std::decay_t<Arg>>) and
      (covariance<Arg> and not square_root_covariance<Arg>) and
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


    /// Assign from a compatible square \ref typed_matrix (assumed, without checking, to be self-adjoint).
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires square_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, NestedSelfAdjoint>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_matrix<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      modifiable<NestedMatrix, NestedSelfAdjoint>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        std::scoped_lock lock {nested_mutex};
        Base::operator=(internal::to_covariance_nestable<NestedSelfAdjoint>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /// Assign from a compatible \ref covariance_nestable.
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires modifiable<NestedMatrix, Arg>
#else
    template<typename Arg, std::enable_if_t<(covariance_nestable<Arg>) and modifiable<NestedMatrix, Arg>, int> = 0>
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


    /// Assign from a compatible \ref typed_matrix_nestable (assumed, without checking, to be self-adjoint).
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
        std::scoped_lock lock {nested_mutex};
        Base::operator=(internal::to_covariance_nestable<NestedSelfAdjoint>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Increment by another non-square-root \ref covariance or square \ref typed_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((covariance<Arg> and not square_root_covariance<Arg>) or (typed_matrix<Arg> and square_matrix<Arg>)) and
        equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((covariance<Arg> and not square_root_covariance<Arg>) or (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        nested_matrix() += internal::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0)
        {
          decltype(auto) E1 = nested_matrix();
          decltype(auto) E2 = internal::to_covariance_nestable<NestedMatrix>(arg);
          if constexpr(upper_triangular_matrix<NestedMatrix>)
            nested_matrix() = QR_decomposition(concatenate_vertical(E1, E2));
          else
            nested_matrix() = LQ_decomposition(concatenate_horizontal(E1, E2));
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() += internal::to_covariance_nestable<NestedSelfAdjoint>(arg);
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
    ((covariance<Arg> and not square_root_covariance<Arg>) or (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((covariance<Arg> and not square_root_covariance<Arg>) or (typed_matrix<Arg> and square_matrix<Arg>)) and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      std::scoped_lock lock {nested_mutex};
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        nested_matrix() -= internal::to_covariance_nestable<NestedMatrix>(arg);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0)
        {
          using TLowerType = typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<TriangleType::lower>;
          const auto U = internal::to_covariance_nestable<TLowerType>(arg);
          OpenKalman::rank_update(nested_matrix(), U, Scalar(-1));
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() -= internal::to_covariance_nestable<NestedSelfAdjoint>(arg);
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
      std::scoped_lock lock {nested_mutex};
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        nested_matrix() *= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else if (s > 0)
      {
        if (synchronization_direction() >= 0) nested_matrix() *= std::sqrt(static_cast<const Scalar>(s));
        if (synchronization_direction() <= 0) cholesky_nested_matrix() *= static_cast<const Scalar>(s);
      }
      else if (s < 0)
      {
        if (synchronization_direction() >= 0)
        {
          using TLowerType = typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<TriangleType::lower>;
          const auto U = internal::to_covariance_nestable<TLowerType>(*this);
          nested_matrix() = MatrixTraits<NestedMatrix>::zero();
          OpenKalman::rank_update(nested_matrix(), U, s);
        }
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() *= static_cast<const Scalar>(s);
        }
      }
      else
      {
        nested_matrix() = MatrixTraits<NestedMatrix>::zero();
        if (synchronization_direction() <= 0)
        {
          cholesky_nested_matrix() = MatrixTraits<NestedSelfAdjoint>::zero();
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
      std::scoped_lock lock {nested_mutex};
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        nested_matrix() /= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else if (s > 0)
      {
        if (synchronization_direction() >= 0) nested_matrix() /= std::sqrt(static_cast<const Scalar>(s));
        if (synchronization_direction() <= 0) cholesky_nested_matrix() /= static_cast<const Scalar>(s);
      }
      else if (s < 0)
      {
        if (synchronization_direction() >= 0)
        {
          using TLowerType = typename MatrixTraits<NestedMatrix>::template TriangularMatrixFrom<TriangleType::lower>;
          const auto u = internal::to_covariance_nestable<TLowerType>(*this);
          nested_matrix() = MatrixTraits<NestedMatrix>::zero();
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
      std::scoped_lock lock {nested_mutex};
      if constexpr(self_adjoint_matrix<NestedMatrix>)
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
      std::scoped_lock lock {nested_mutex};
      if constexpr(self_adjoint_matrix<NestedMatrix>)
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


    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }


    /**
     * \brief Take the Cholesky square root of *this.
     * \details If *this is an lvalue reference, this creates a reference to the nested matrix rather than a copy.
     * \return A SquareRootCovariance based on *this.
     */
#ifdef __cpp_concepts
    auto square_root() & requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square_root() &
#endif
    {
      return SquareRootCovariance<Coefficients, std::add_lvalue_reference_t<NestedMatrix>>(*this);
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square_root() && requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square_root() &&
#endif
    {
      return SquareRootCovariance<Coefficients, self_contained_t<NestedMatrix>>(std::move(*this));
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square_root() const & requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square_root() const &
#endif
    {
      return SquareRootCovariance<Coefficients, std::add_lvalue_reference_t<const NestedMatrix>>(*this);
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square_root() const && requires (not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>
#else
    template<typename T = NestedMatrix, std::enable_if_t<(not diagonal_matrix<T>) or identity_matrix<NestedMatrix> or
      zero_matrix<NestedMatrix>, int> = 0>
    auto square_root() const &&
#endif
    {
      return SquareRootCovariance<Coefficients, const self_contained_t<NestedMatrix>>(std::move(*this));
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square_root() const & requires diagonal_matrix<NestedMatrix> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<diagonal_matrix<T> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>), int> = 0>
    auto square_root() const &
#endif
    {
      auto n = make_self_contained(Cholesky_factor<storage_triangle>(nested_matrix()));
      return SquareRootCovariance<Coefficients, decltype(n)> {std::move(n)};
    }


    /**
     * \overload
     * \details This overload is operative if the matrix is diagonal.
     */
#ifdef __cpp_concepts
    auto square_root() const && requires diagonal_matrix<NestedMatrix> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>)
#else
    template<typename T = NestedMatrix, std::enable_if_t<diagonal_matrix<T> and (not identity_matrix<NestedMatrix>) and
      (not zero_matrix<NestedMatrix>), int> = 0>
    auto square_root() const &&
#endif
    {
      constexpr decltype(auto) fw = [] (auto&& n) { return std::forward<decltype(n)>(n); };
      auto n = make_self_contained(Cholesky_factor<storage_triangle>(fw(nested_matrix())));
      return SquareRootCovariance<Coefficients, decltype(n)>{std::move(n)};
    }


    /**
     * \internal
     * \brief Make a Covariance based on an operation on the nested matrices.
     * \tparam F1 Operation on NestedMatrix.
     * \tparam F2 Operation on the return value of cholesky_nested_matrix
     */
#ifdef __cpp_concepts
    template<typename F1, typename F2> requires
      std::invocable<F1, const NestedMatrix&> and std::invocable<F2, const NestedSelfAdjoint&>
#else
    template<typename F1, typename F2, std::enable_if_t<
      std::is_invocable_v<F1, const NestedMatrix&> and std::is_invocable_v<F2, const NestedSelfAdjoint&>, int> = 0>
#endif
    auto covariance_op(F1&& f1, F2&& f2) const
    {
      auto n = make_self_contained(f1(nested_matrix()));
      using N = decltype(n);
      if constexpr (internal::case1or2<Covariance, N>)
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
          return Covariance<Coefficients, N> {internal::to_covariance_nestable<N>(f2(cholesky_nested_matrix()))};
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
    template<typed_matrix U> requires equivalent_to<typename MatrixTraits<U>::RowCoefficients, Coefficients>
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      equivalent_to<typename MatrixTraits<U>::RowCoefficients, Coefficients>, int> = 0>
#endif
    auto rank_update(const U& u, const Scalar alpha = 1) const
    {
      std::scoped_lock lock {nested_mutex};
      if (synchronization_direction() < 0) synchronize_reverse();
      if constexpr (one_by_one_matrix<NestedMatrix>)
      {
        auto b = trace(nested_matrix()) + alpha * trace(u * adjoint(u));
        return make(MatrixTraits<NestedMatrix>::make(b));
      }
      else
      {
        return make(OpenKalman::rank_update(nested_matrix(), OpenKalman::nested_matrix(u), alpha));
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
    (C::size == MatrixTraits<N>::rows) and (not std::is_rvalue_reference_v<N>)
#else
    template<typename, typename>
#endif
    friend struct Covariance;


#ifdef __cpp_concepts
    template<coefficients C, covariance_nestable N> requires
      (C::size == MatrixTraits<N>::rows) and (not std::is_rvalue_reference_v<N>)
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
  explicit Covariance(M&&) -> Covariance<Axes<MatrixTraits<M>::rows>, passable_t<M>>;


  /**
   * \brief Deduce Covariance type from a square \ref typed_matrix
   */
#ifdef __cpp_concepts
  template<typed_matrix M> requires square_matrix<M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M> and square_matrix<M>, int> = 0>
#endif
  explicit Covariance(M&&) -> Covariance<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<nested_matrix_t<M>>::template SelfAdjointMatrixFrom<>>;


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
    Axes<MatrixTraits<M>::rows>, typename MatrixTraits<M>::template SelfAdjointMatrixFrom<>>;


  // ---------------- //
  //  Make Functions  //
  // ---------------- //

  /**
   * \brief Make a Covariance from a \ref covariance_nestable, specifying the coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable with size matching Coefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable Arg> requires
    (Coefficients::size == MatrixTraits<Arg>::rows)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and
    covariance_nestable<Arg> and (Coefficients::size == MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    return Covariance<Coefficients, passable_t<Arg>> {std::forward<Arg>(arg)};
  }


  /**
   * \brief Make a Covariance from a \ref covariance_nestable, specifying the coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower, diagonal).
   * \tparam Arg A \ref covariance_nestable with size matching Coefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, covariance_nestable Arg> requires
    (Coefficients::size == MatrixTraits<Arg>::rows) and
    (triangle_type != TriangleType::lower or lower_triangular_matrix<Arg>) and
    (triangle_type != TriangleType::upper or upper_triangular_matrix<Arg>) and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<Arg>)
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and covariance_nestable<Arg> and
    (Coefficients::size == MatrixTraits<Arg>::rows) and
    (triangle_type != TriangleType::lower or lower_triangular_matrix<Arg>) and
    (triangle_type != TriangleType::upper or upper_triangular_matrix<Arg>) and
    (triangle_type != TriangleType::diagonal or diagonal_matrix<Arg>), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    return Covariance<Coefficients, passable_t<Arg>> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   * \brief Make a Covariance from a \ref covariance_nestable, with default Axis coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
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
    using C = Axes<MatrixTraits<Arg>::rows>;
    return make_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a Covariance (with nested triangular matrix) from a self-adjoint \ref typed_matrix_nestable.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable with size matching Coefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, typed_matrix_nestable Arg> requires
    (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (Coefficients::size == MatrixTraits<Arg>::rows) and (Coefficients::size == MatrixTraits<Arg>::columns)
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (Coefficients::size == MatrixTraits<Arg>::rows) and
    (Coefficients::size == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using T = typename MatrixTraits<Arg>::template TriangularMatrixFrom<triangle_type>;
    return Covariance<Coefficients, T> {std::forward<Arg>(arg)};
  }


  /**
   * \overload
   * \brief Make a Covariance from a self-adjoint \ref typed_matrix_nestable, specifying the coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam Arg A square \ref typed_matrix_nestable with size matching Coefficients.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and
    (Coefficients::size == MatrixTraits<Arg>::rows) and (Coefficients::size == MatrixTraits<Arg>::columns)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (Coefficients::size == MatrixTraits<Arg>::rows) and
    (Coefficients::size == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using SA = typename MatrixTraits<Arg>::template SelfAdjointMatrixFrom<>;
    return make_covariance<Coefficients, SA>(internal::to_covariance_nestable<SA>(std::forward<Arg>(arg)));
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
    using C = Axes<MatrixTraits<Arg>::rows>;
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
    using C = Axes<MatrixTraits<Arg>::rows>;
    return make_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance from a \ref covariance_nestable or \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename Arg> requires
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    constexpr TriangleType triangle_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularMatrixFrom<>>;
    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalMatrixFrom<>,
      std::conditional_t<triangular_matrix<Arg>,
        typename MatrixTraits<Arg>::template TriangularMatrixFrom<triangle_type>,
        typename MatrixTraits<Arg>::template SelfAdjointMatrixFrom<triangle_type>>>;
    return Covariance<Coefficients, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance with a nested triangular matrix, from a \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, typed_matrix_nestable Arg> requires
    square_matrix<Arg>
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<coefficients<Coefficients> and typed_matrix_nestable<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using B = std::conditional_t<triangle_type == TriangleType::diagonal,
      typename MatrixTraits<Arg>::template DiagonalMatrixFrom<>,
      typename MatrixTraits<Arg>::template TriangularMatrixFrom<triangle_type>>;
    return Covariance<Coefficients, B>();
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
    using C = Axes<MatrixTraits<Arg>::rows>;
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
    using C = Axes<MatrixTraits<Arg>::rows>;
    return make_covariance<C, triangle_type, Arg>();
  }


  /**
   * \overload
   * \brief Make a Covariance based on another non-square-root \ref covariance.
   */
#ifdef __cpp_concepts
  template<covariance Arg> requires (not square_root_covariance<Arg>)
#else
  template<typename Arg, std::enable_if_t<covariance<Arg> and (not square_root_covariance<Arg>), int> = 0>
#endif
  inline auto
  make_covariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return Covariance<C, nested_matrix_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized Covariance from a non-square-root \ref covariance.
   */
#ifdef __cpp_concepts
  template<covariance T> requires (not square_root_covariance<T>)
#else
  template<typename T, std::enable_if_t<covariance<T> and (not square_root_covariance<T>), int> = 0>
#endif
  inline auto
  make_covariance()
  {
    using C = typename MatrixTraits<T>::RowCoefficients;
    using B = nested_matrix_t<T>;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
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
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
    return make_covariance<C, B>();
  }


  // --------------------- //
  //        Traits         //
  // --------------------- //

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<Covariance<Coeffs, ArgType>>
  {
    using NestedMatrix = ArgType;
    static constexpr auto rows = MatrixTraits<NestedMatrix>::rows;
    static constexpr auto columns = rows;
    using RowCoefficients = Coeffs;
    using ColumnCoefficients = Coeffs;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

    template<std::size_t r = rows, std::size_t c = rows, typename S = Scalar>
    using NativeMatrixFrom = native_matrix_t<NestedMatrix, r, c, S>;

    using SelfContainedFrom = Covariance<Coeffs, self_contained_t<NestedMatrix>>;

#ifdef __cpp_concepts
    template<coefficients C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg, std::enable_if_t<coefficients<C> and covariance_nestable<Arg>,int> = 0>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return Covariance<Coeffs, std::decay_t<NestedMatrix>>::zero(); }

    static auto identity() { return Covariance<Coeffs, std::decay_t<NestedMatrix>>::identity(); }
  };


} // OpenKalman

#endif //OPENKALMAN_COVARIANCE_HPP
