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

#include "basics/basics.hpp"

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  // ---------------------- //
  //  SquareRootCovariance  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, covariance_nestable NestedMatrix> requires
    (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and values::number<scalar_type_of_t<NestedMatrix>>
#else
  template<typename StaticDescriptor, typename NestedMatrix>
#endif
  struct SquareRootCovariance : oin::CovarianceImpl<SquareRootCovariance<StaticDescriptor, NestedMatrix>, NestedMatrix>
  {

#ifndef __cpp_concepts
    static_assert(fixed_pattern<StaticDescriptor>);
    static_assert(covariance_nestable<NestedMatrix>);
    static_assert(coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<NestedMatrix, 0>);
    static_assert(not std::is_rvalue_reference_v<NestedMatrix>);
    static_assert(values::number<scalar_type_of_t<NestedMatrix>>);
#endif

    // May be accessed externally through MatrixTraits:
    using Scalar = scalar_type_of_t<NestedMatrix>; ///< Scalar type for this matrix.

  private:

    using Base = oin::CovarianceImpl<SquareRootCovariance, NestedMatrix>;
    using typename Base::CholeskyNestedMatrix;
    using Base::nested_object;
    using Base::cholesky_nested_matrix;
    using Base::synchronization_direction;
    using Base::synchronize_forward;
    using Base::synchronize_reverse;
    using Base::mark_nested_matrix_changed;
    using Base::mark_cholesky_nested_matrix_changed;
    using Base::mark_synchronized;


    // May be accessed externally through MatrixTraits:
    static constexpr auto dim = index_dimension_of_v<NestedMatrix, 0>;

    // May be accessed externally through MatrixTraits:
    static constexpr TriangleType triangle_type =
      triangle_type_of_v<typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularAdapterFrom<>>;

    // A triangular nested matrix type.
    using NestedTriangular = std::conditional_t<triangular_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<std::decay_t<NestedMatrix>>::template TriangularAdapterFrom<triangle_type>>;


    // A function that makes a self-contained covariance from a nested matrix.
    template<typename C = StaticDescriptor, typename Arg>
    static auto make(Arg&& arg)
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }


    /**
     * \brief Construct from a non-square-root, non-diagonal \ref covariance.
     */
#ifdef __cpp_concepts
    template<self_adjoint_covariance M> requires (not diagonal_matrix<M> or identity_matrix<M> or zero<M>) and
      (hermitian_matrix<nested_object_of_t<M>> == hermitian_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<self_adjoint_covariance<M> and
      (not diagonal_matrix<M> or identity_matrix<M> or zero<M>) and
      (hermitian_matrix<nested_object_of_t<M>> == hermitian_matrix<NestedMatrix>), int> = 0>
#endif
    SquareRootCovariance(M&& m) : Base {std::forward<M>(m)} {}


  public:
    // ------------ //
    // Constructors //
    // ------------ //

    /// Default constructor.
#ifdef __cpp_concepts
    SquareRootCovariance() requires std::default_initializable<Base>
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdcompat::default_initializable<Base>, int> = 0>
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
      stdcompat::constructible_from<Base, M&&>, int> = 0>
#endif
    SquareRootCovariance(M&& m) : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable M> requires requires(M&& m) { Base {std::forward<M>(m)}; }
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      stdcompat::constructible_from<Base, M&&>, int> = 0>
#endif
    explicit SquareRootCovariance(M&& m) : Base {std::forward<M>(m)} {}


    /**
     * \brief Construct from a \ref typed_matrix.
     * \details M must be a \ref square_shaped, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be triangular, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix M> requires (square_shaped<M> or (diagonal_matrix<NestedMatrix> and vector<M>)) and
      compares_with<vector_space_descriptor_of_t<M, 0>, StaticDescriptor> and
      requires(M&& m) { Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))}; }
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and
      (square_shaped<M> or (diagonal_matrix<NestedMatrix> and vector<M>)) and
      compares_with<vector_space_descriptor_of_t<M, 0>, StaticDescriptor> and
      stdcompat::constructible_from<Base,
        decltype(oin::to_covariance_nestable<NestedTriangular>(std::declval<M&&>()))>, int> = 0>
#endif
    explicit SquareRootCovariance(M&& m)
      : Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))} {}


    /**
     * \brief Construct from a \ref typed_matrix_nestable.
     * \details M must be a \ref square_shaped, unless NestedMatrix is a \ref diagonal_matrix in which case M can be
     * a column vector.
     * M is assumed (without enforcement) to be triangular, and the data in only one of the triangles is significant.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and
      (square_shaped<M> or (diagonal_matrix<NestedMatrix> and vector<M>)) and
      requires(M&& m) { Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))}; }
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and (not covariance_nestable<M>) and
      (square_shaped<M> or (diagonal_matrix<NestedMatrix> and vector<M>)) and
      stdcompat::constructible_from<Base,
        decltype(oin::to_covariance_nestable<NestedTriangular>(std::declval<M&&>()))>, int> = 0>
#endif
    explicit SquareRootCovariance(M&& m)
      : Base {oin::to_covariance_nestable<NestedTriangular>(std::forward<M>(m))} {}


    /// Construct from Scalar coefficients. Assumes matrix is triangular, and only reads lower left triangle.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> ... Args> requires (sizeof...(Args) > 0) and
      requires(Args ... args) { Base {make_dense_object_from<NestedTriangular>(static_cast<const Scalar>(args)...)};
      }
#else
    template<typename ... Args, std::enable_if_t<(stdcompat::convertible_to<Args, const Scalar> and ...) and
      ((diagonal_matrix<NestedMatrix> and sizeof...(Args) == dim) or
        (sizeof...(Args) == dim * dim)) and stdcompat::constructible_from<Base, NestedTriangular&&>, int> = 0>
#endif
    SquareRootCovariance(Args ... args)
      : Base {make_dense_object_from<NestedTriangular>(static_cast<const Scalar>(args)...)} {}


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
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor> and
      std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<triangular_covariance<Arg> and
      (not std::is_base_of_v<SquareRootCovariance, std::decay_t<Arg>>) and
      (triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(other));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref typed_matrix (assumed, without checking, to be triangular).
     */
#ifdef __cpp_concepts
    template<typed_matrix Arg> requires square_shaped<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor> and
      std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, NestedTriangular>
#else
    template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_shaped<Arg> and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, NestedTriangular>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(oin::to_covariance_nestable<NestedTriangular>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Arg> requires std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
      {
        Base::operator=(std::forward<Arg>(other));
      }
      return *this;
    }


    /**
     * \brief Assign from a compatible \ref typed_matrix_nestable (assumed, without checking, to be triangular).
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable Arg> requires (not covariance_nestable<Arg>) and square_shaped<Arg> and
      std::assignable_from<std::add_lvalue_reference_t<NestedMatrix>, NestedTriangular>
#else
    template<typename Arg, std::enable_if_t<typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
      square_shaped<Arg> and std::is_assignable_v<std::add_lvalue_reference_t<NestedMatrix>, NestedTriangular>, int> = 0>
#endif
    auto& operator=(Arg&& other)
    {
      if constexpr (not zero<NestedMatrix> and not identity_matrix<NestedMatrix>)
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
        (typed_matrix<Arg> and square_shaped<Arg>)) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((triangular_covariance<Arg> and triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) or
        (typed_matrix<Arg> and square_shaped<Arg>)) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor>, int> = 0>
#endif
    auto& operator+=(const Arg& arg)
    {
      if constexpr(triangular_matrix<NestedMatrix>) // Case 1 or 2
      {
        nested_object() += oin::to_covariance_nestable<NestedMatrix>(arg);
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
      (typed_matrix<Arg> and square_shaped<Arg>)) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor>
#else
    template<typename Arg, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<NestedMatrix>>) and
      ((triangular_covariance<Arg> and triangle_type_of<Arg>::value == triangle_type_of<SquareRootCovariance>::value) or
        (typed_matrix<Arg> and square_shaped<Arg>)) and
      compares_with<vector_space_descriptor_of_t<Arg, 0>, StaticDescriptor>, int> = 0>
#endif
    auto& operator-=(const Arg& arg)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_object() -= oin::to_covariance_nestable<NestedMatrix>(arg);
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
    template<typename S, std::enable_if_t<stdcompat::convertible_to<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_object() *= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0) nested_object() *= static_cast<const Scalar>(s) * s;
        if (synchronization_direction() <= 0) cholesky_nested_matrix() *= static_cast<const Scalar>(s);
      }
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S> requires (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename S, std::enable_if_t<stdcompat::convertible_to<S, Scalar> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& operator/=(const S s)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_object() /= static_cast<const Scalar>(s);
        mark_nested_matrix_changed();
      }
      else
      {
        if (synchronization_direction() >= 0) nested_object() /= static_cast<const Scalar>(s) * s;
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
        nested_object() *= oin::to_covariance_nestable<NestedMatrix>(arg);
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
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero<NestedMatrix>)
      {
        return Covariance<StaticDescriptor, std::remove_reference_t<NestedMatrix>&> {*this};
      }
      else
      {
        auto n = cholesky_square(nested_object());
        return Covariance<StaticDescriptor, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square() const &
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero<NestedMatrix>)
      {
        return Covariance<StaticDescriptor, const std::remove_reference_t<NestedMatrix>&> {*this};
      }
      else
      {
        auto n = cholesky_square(nested_object());
        return Covariance<StaticDescriptor, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square() &&
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero<NestedMatrix>)
      {
        return Covariance<StaticDescriptor, std::remove_reference_t<NestedMatrix>> {std::move(*this)};
      }
      else
      {
        auto n = cholesky_square(std::move(*this).nested_object());
        return Covariance<StaticDescriptor, decltype(n)> {std::move(n)};
      }
    }


    /// \overload
    auto square() const &&
    {
      if constexpr ((not diagonal_matrix<NestedMatrix>) or identity_matrix<NestedMatrix> or zero<NestedMatrix>)
      {
        return Covariance<StaticDescriptor, std::remove_reference_t<NestedMatrix>> {std::move(*this)};
      }
      else
      {
        auto n = cholesky_square(std::move(*this).nested_object());
        return Covariance<StaticDescriptor, decltype(n)> {std::move(n)};
      }
    }


    /**
     * \brief Perform a rank update.
     */
#ifdef __cpp_concepts
    template<typed_matrix U> requires compares_with<vector_space_descriptor_of_t<U, 0>, StaticDescriptor> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>)
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      compares_with<vector_space_descriptor_of_t<U, 0>, StaticDescriptor> and
      (not std::is_const_v<std::remove_reference_t<NestedMatrix>>), int> = 0>
#endif
    auto& rank_update(const U& u, const Scalar alpha = 1) &
    {
      if (synchronization_direction() < 0) synchronize_reverse();
      OpenKalman::rank_update(nested_object(), OpenKalman::nested_object(u), alpha);
      mark_nested_matrix_changed();
      return *this;
    }


    /**
     * \brief Perform a rank update.
     */
#ifdef __cpp_concepts
    template<typed_matrix U> requires compares_with<vector_space_descriptor_of_t<U, 0>, StaticDescriptor>
#else
    template<typename U, std::enable_if_t<typed_matrix<U> and
      compares_with<vector_space_descriptor_of_t<U, 0>, StaticDescriptor>, int> = 0>
#endif
    auto rank_update(const U& u, const Scalar alpha = 1) &&
    {
      if (synchronization_direction() < 0) synchronize_reverse();
      return make(OpenKalman::rank_update(nested_object(), OpenKalman::nested_object(u), alpha));
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
    template<fixed_pattern C, covariance_nestable N> requires
    (coordinates::dimension_of_v<C> == index_dimension_of_v<N, 0>) and (not std::is_rvalue_reference_v<N>)
#else
    template<typename, typename>
#endif
    friend struct SquareRootCovariance;


#ifdef __cpp_concepts
    template<fixed_pattern C, covariance_nestable N> requires
      (coordinates::dimension_of_v<C> == index_dimension_of_v<N, 0>) and (not std::is_rvalue_reference_v<N>)
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
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<Dimensions<index_dimension_of_v<M, 0>>, passable_t<M>>;


  /**
   * \brief Deduce SquareRootCovariance type from a square \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix M> requires square_shaped<M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M> and square_shaped<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<vector_space_descriptor_of_t<M, 0>,
    typename MatrixTraits<std::decay_t<nested_object_of_t<M>>>::template TriangularAdapterFrom<>>;


  /**
   * \brief Deduce SquareRootCovariance type from a square \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<typed_matrix_nestable M> requires (not covariance_nestable<M>) and square_shaped<M>
#else
  template<typename M, std::enable_if_t<
    typed_matrix_nestable<M> and (not covariance_nestable<M>) and square_shaped<M>, int> = 0>
#endif
  explicit SquareRootCovariance(M&&) -> SquareRootCovariance<Dimensions<index_dimension_of_v<M, 0>>,
    typename MatrixTraits<std::decay_t<M>>::template TriangularAdapterFrom<>>;


  // ---------------- //
  //  Make Functions  //
  // ---------------- //

  /**
   * \brief Make a SquareRootCovariance from a \ref covariance_nestable, specifying the coefficients.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable with size matching StaticDescriptor.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, covariance_nestable Arg> requires
    (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<Arg, 0>)
#else
  template<typename StaticDescriptor, typename Arg, std::enable_if_t<fixed_pattern<StaticDescriptor> and
    covariance_nestable<Arg> and (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of<Arg, 0>::value), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg)
  {
    return SquareRootCovariance<StaticDescriptor, passable_t<Arg>>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance from a \ref covariance_nestable, with default Axis coefficients.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam Arg A \ref covariance_nestable.
   */
#ifdef __cpp_concepts
  template<covariance_nestable Arg>
#else
  template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg)
  {
    using C = Dimensions<index_dimension_of_v<Arg, 0>>;
    return make_square_root_covariance<C>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance (with nested triangular matrix) from a self-adjoint \ref typed_matrix_nestable.
   * \tparam StaticDescriptor The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint \ref typed_matrix_nestable with size matching StaticDescriptor.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, TriangleType triangle_type = TriangleType::lower, typed_matrix_nestable Arg>
  requires (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<Arg, 0>) and (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of_v<Arg, 1>)
#else
  template<typename StaticDescriptor, TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<
    fixed_pattern<StaticDescriptor> and typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of<Arg, 0>::value) and
    (coordinates::dimension_of_v<StaticDescriptor> == index_dimension_of<Arg, 1>::value), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg)
  {
    using T = typename MatrixTraits<std::decay_t<Arg>>::template TriangularAdapterFrom<triangle_type>;
    return SquareRootCovariance<StaticDescriptor, T>(std::forward<Arg>(arg));
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
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_shaped<Arg>
#else
  template<TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<
    typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg)
  {
    using C = Dimensions<index_dimension_of_v<Arg, 0>>;
    return make_square_root_covariance<C, triangle_type>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, TriangleType triangle_type, typed_matrix_nestable Arg> requires square_shaped<Arg>
#else
  template<typename StaticDescriptor, TriangleType triangle_type, typename Arg,
    std::enable_if_t<fixed_pattern<StaticDescriptor> and typed_matrix_nestable<Arg> and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using B = std::conditional_t<triangle_type == TriangleType::diagonal,
    typename MatrixTraits<std::decay_t<Arg>>::template DiagonalMatrixFrom<>,
      typename MatrixTraits<std::decay_t<Arg>>::template TriangularAdapterFrom<triangle_type>>;
    return SquareRootCovariance<StaticDescriptor, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a \ref covariance_nestable or \ref typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<fixed_pattern StaticDescriptor, typename Arg> requires
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_shaped<Arg>
#else
  template<typename StaticDescriptor, typename Arg, std::enable_if_t<
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    constexpr TriangleType template_type = triangle_type_of_v<typename MatrixTraits<std::decay_t<Arg>>::template TriangularAdapterFrom<>>;
    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<std::decay_t<Arg>>::template DiagonalMatrixFrom<>,
      std::conditional_t<hermitian_matrix<Arg>,
        typename MatrixTraits<std::decay_t<Arg>>::template SelfAdjointMatrixFrom<template_type>,
        typename MatrixTraits<std::decay_t<Arg>>::template TriangularAdapterFrom<template_type>>>;
    return SquareRootCovariance<StaticDescriptor, B>();
  }


/**
 * \overload
 * \brief Make a writable, uninitialized SquareRootCovariance from a \ref typed_matrix_nestable or \ref covariance_nestable.
 * \details The coefficients will be Axis.
 */
#ifdef __cpp_concepts
  template<typename Arg> requires (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and square_shaped<Arg>
#else
  template<typename Arg, std::enable_if_t<(covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = Dimensions<index_dimension_of_v<Arg, 0>>;
    return make_square_root_covariance<C, Arg>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance, with default Axis coefficients.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix_nestable Arg> requires square_shaped<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<
    typed_matrix_nestable<Arg> and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = Dimensions<index_dimension_of_v<Arg, 0>>;
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
  make_square_root_covariance(Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    return SquareRootCovariance<C, nested_object_of_t<Arg>>(std::forward<Arg>(arg));
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
    using C = vector_space_descriptor_of_t<Arg, 0>;
    using B = nested_object_of_t<Arg>;
    return make_square_root_covariance<C, B>();
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance from a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type = TriangleType::lower, typed_matrix Arg> requires
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_shaped<Arg>
#else
  template<TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    return make_square_root_covariance<C, triangle_type>(nested_object(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance based on a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg> requires square_shaped<Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<
    typed_matrix<Arg> and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    using B = nested_object_of_t<Arg>;
    return make_square_root_covariance<C, triangle_type, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance based on a \ref typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires square_shaped<Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and square_shaped<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    using B = nested_object_of_t<Arg>;
    return make_square_root_covariance<C, B>();
  }


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Coeffs, typename NestedMatrix>
    struct indexible_object_traits<SquareRootCovariance<Coeffs, NestedMatrix>>
    {
      using scalar_type = scalar_type_of_t<NestedMatrix>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 2>{}; }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, N)
      {
        return std::forward<Arg>(arg).my_dimension;
      }


      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        if constexpr (hermitian_matrix<NestedMatrix>)
          return std::forward<Arg>(arg).get_self_adjoint_nested_matrix();
        else
          return std::forward<Arg>(arg).get_triangular_nested_matrix();
      }


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg)
      {
        if constexpr (zero<NestedMatrix>)
          return constant_coefficient{arg.nestedExpression()};
        else
          return std::monostate {};
      }


      template<typename Arg>
      static constexpr auto get_constant_diagonal(const Arg& arg)
      {
        return constant_diagonal_coefficient {arg.nestedExpression()};
      }


      template<Applicability b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<NestedMatrix, b>;


      template<Applicability b>
      static constexpr bool is_square = true;


      template<TriangleType t>
      static constexpr bool is_triangular = triangular_matrix<NestedMatrix, t> or
        hermitian_adapter<NestedMatrix, t == TriangleType::upper ? HermitianAdapterType::upper : HermitianAdapterType::lower>;


      static constexpr bool is_triangular_adapter = false;


      static constexpr bool is_hermitian = false;


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires
        element_gettable<decltype(std::declval<Arg&&>().get_triangular_nested_matrix()), sizeof...(I)>
  #else
      template<typename Arg, typename...I, std::enable_if_t<
        element_gettable<decltype(std::declval<Arg&&>().get_triangular_nested_matrix()), sizeof...(I)>, int> = 0>
  #endif
      static constexpr auto get(Arg&& arg, I...i)
      {
        return std::forward<Arg>(arg)(i...);
      }


  #ifdef __cpp_lib_concepts
      template<typename Arg, typename...I> requires
        writable_by_component<decltype(std::declval<Arg&>().get_triangular_nested_matrix()), sizeof...(I)>
  #else
      template<typename Arg, typename...I, std::enable_if_t<
        writable_by_component<decltype(std::declval<Arg&>().get_triangular_nested_matrix()), sizeof...(I)>, int> = 0>
  #endif
      static constexpr void set(Arg& arg, const scalar_type_of_t<Arg>& s, I...i)
      {
        arg.set_component(s, i...);
      }

      static constexpr bool is_writable = library_interface<std::decay_t<NestedMatrix>>::is_writable;


#ifdef __cpp_lib_concepts
      template<typename Arg> requires one_dimensional<NestedMatrix> and raw_data_defined_for<NestedMatrix>
#else
      template<typename Arg, std::enable_if_t<one_dimensional<NestedMatrix> and raw_data_defined_for<NestedMatrix>, int> = 0>
#endif
      static constexpr auto * const
      raw_data(Arg& arg) { return internal::raw_data(arg.nested_object()); }


      static constexpr Layout layout = one_dimensional<NestedMatrix> ? layout_of_v<NestedMatrix> : Layout::none;

    };

  } // namespace interface


}


#endif //OPENKALMAN_SQUAREROOTCOVARIANCE_HPP

