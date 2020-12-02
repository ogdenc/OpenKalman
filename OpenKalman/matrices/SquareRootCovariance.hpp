/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SQUAREROOTCOVARIANCE_HPP
#define OPENKALMAN_SQUAREROOTCOVARIANCE_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<coefficients Coeffs, covariance_nestable ArgType> requires
    (Coeffs::size == MatrixTraits<ArgType>::dimension) and (not std::is_rvalue_reference_v<ArgType>)
#else
  template<typename Coeffs, typename ArgType>
#endif
  struct SquareRootCovariance : internal::CovarianceBase<SquareRootCovariance<Coeffs, ArgType>, ArgType>
  {
    static_assert(Coeffs::size == MatrixTraits<ArgType>::dimension);
    static_assert(not std::is_rvalue_reference_v<ArgType>);
    using NestedMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    using Base = internal::CovarianceBase<SquareRootCovariance, ArgType>;
    static constexpr TriangleType triangle_type =
      triangle_type_of<typename MatrixTraits<NestedMatrix>::template TriangularBaseType<>>;

  protected:
    using TBaseType = std::conditional_t<diagonal_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<NestedMatrix>::template TriangularBaseType<triangle_type>>;

    template<typename C = Coefficients, typename Arg>
    static constexpr auto
    make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, self_contained_t<Arg>>(std::forward<Arg>(arg));
    }

  public:
    /**************
     * Constructors
     **************/

    /// Default constructor.
    SquareRootCovariance() : Base() {}


    /// Copy constructor.
    SquareRootCovariance(const SquareRootCovariance& other) : Base(other) {}


    /// Move constructor.
    SquareRootCovariance(SquareRootCovariance&& other) : Base(std::move(other)) {}


    /// Construct from a general covariance type.
#ifdef __cpp_concepts
    template<covariance M> requires
      (not (diagonal_matrix<M> and not square_root_covariance<M> and diagonal_matrix<NestedMatrix>))
#else
    template<typename M, std::enable_if_t<covariance<M> and
      (not (diagonal_matrix<M> and not square_root_covariance<M> and diagonal_matrix<NestedMatrix>)), int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(std::forward<M>(m))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      using MBase = nested_matrix_t<M>;
      static_assert(not square_root_covariance<M> or self_adjoint_matrix<MBase> or self_adjoint_matrix<NestedMatrix> or
          internal::same_triangle_type_as<NestedMatrix, MBase>,
        "An upper-triangle Cholesky-form covariance cannot be constructed from a lower-triangle Cholesky-form "
        "covariance, and vice versa. To convert, use adjoint().");
    }


    /// Construct from a general covariance type.
#ifdef __cpp_concepts
    template<covariance M> requires
      diagonal_matrix<M> and (not square_root_covariance<M>) and diagonal_matrix<NestedMatrix>
#else
    template<typename M, std::enable_if_t<covariance<M> and
      diagonal_matrix<M> and not square_root_covariance<M> and diagonal_matrix<NestedMatrix>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(Cholesky_factor(std::forward<M>(m).nested_matrix()))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
    }


    /// Construct from a non-diagonal covariance_nestable.
#ifdef __cpp_concepts
    template<covariance_nestable M> requires (not diagonal_matrix<M> or triangular_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      (not diagonal_matrix<M> or triangular_matrix<NestedMatrix>), int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(std::forward<M>(m)) {}


    /// Construct from a diagonal covariance_nestable.
#ifdef __cpp_concepts
    template<covariance_nestable M> requires diagonal_matrix<M> and (not triangular_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      diagonal_matrix<M> and not triangular_matrix<NestedMatrix>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(Cholesky_square(std::forward<M>(m))) {}


    /// Construct from a typed matrix (assumed to be triangular).
#ifdef __cpp_concepts
    template<typed_matrix M>
#else
    template<typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept
      : Base(MatrixTraits<TBaseType>::make(OpenKalman::nested_matrix(std::forward<M>(m))))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      if constexpr(diagonal_matrix<NestedMatrix>)
        static_assert(MatrixTraits<M>::columns == 1);
      else
        static_assert(equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, Coefficients>);
    }


    /// Construct from a typed_matrix_nestable (assumed to be triangular).
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>)
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and not covariance_nestable<M>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(MatrixTraits<TBaseType>::make(std::forward<M>(m))) {}


    /// Construct from Scalar coefficients. Assumes matrix is triangular, and only reads lower left triangle.
    template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    SquareRootCovariance(Args ... args) : Base(MatrixTraits<TBaseType>::make(args...)) {}


    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const SquareRootCovariance& other)
    {
      if constexpr(not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(other);
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(SquareRootCovariance&& other) noexcept
    {
      if constexpr(not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(std::move(other));
      return *this;
    }


    /// Assign from a compatible covariance or typed matrix object.
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or typed_matrix<Arg>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> or typed_matrix<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& other) noexcept
    {
      if constexpr(covariance<Arg>)
      {
        static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      }
      else if constexpr(typed_matrix<Arg>)
      {
        static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
          equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      }
      using ArgBase = nested_matrix_t<Arg>;
      static_assert(not square_root_covariance<Arg> or self_adjoint_matrix<ArgBase> or self_adjoint_matrix<NestedMatrix> or
          internal::same_triangle_type_as<NestedMatrix, ArgBase>,
          "An upper-triangle Cholesky-form covariance cannot be assigned a lower-triangle Cholesky-form "
          "covariance, and vice versa. To convert, use adjoint().");

      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(covariance<Arg> and
        diagonal_matrix<Arg> and not square_root_covariance<Arg> and diagonal_matrix<NestedMatrix>)
      {
        Base::operator=(Cholesky_factor(std::forward<Arg>(other).nested_matrix()));
      }
      else
      {
        Base::operator=(internal::convert_nested_matrix<std::decay_t<NestedMatrix>>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * Increment by another square-root (Cholesky) covariance.
     * \warning This is computationally expensive if the nested matrix is self-adjoint. This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& arg) noexcept
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() += internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        const auto sum = internal::convert_nested_matrix<TBaseType>(*this) +
          internal::convert_nested_matrix<TBaseType>(std::forward<Arg>(arg));
        nested_matrix() = Cholesky_square(sum);
      }
      this->mark_changed();
      return *this;
    }


    /**
     * Increment by another square-root (Cholesky) covariance of the same type.
     * \warning This is computationally expensive if the nested matrix is self-adjoint. This can generally be avoided.
     */
    auto& operator+=(const SquareRootCovariance& arg)
    {
      return operator+=<const SquareRootCovariance&>(arg);
    }


    /**
     * Decrement by another square-root (Cholesky) covariance.
     * \warning This is computationally expensive if the nested matrix is self-adjoint. This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& arg) noexcept
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() -= internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        const auto diff = internal::convert_nested_matrix<TBaseType>(*this) -
          internal::convert_nested_matrix<TBaseType>(std::forward<Arg>(arg));
        nested_matrix() = Cholesky_square(diff);
      }
      this->mark_changed();
      return *this;
    }

    /**
     * Decrement by another square-root (Cholesky) covariance of the same type.
     * \warning This is computationally expensive if the nested matrix is self-adjoint. This can generally be avoided.
     */
    auto& operator-=(const SquareRootCovariance& arg)
    {
      return operator-=<const SquareRootCovariance&>(arg);
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() *= s;
      }
      else
      {
        nested_matrix() *= static_cast<Scalar>(s) * s;
      }
      this->mark_changed();
      return *this;
    }


#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator/=(const S s)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() /= s;
      }
      else
      {
        nested_matrix() /= static_cast<Scalar>(s) * s;
      }
      this->mark_changed();
      return *this;
    }


    /**
     * \brief Multiply by another square-root covariance matrix.
     * \details If the underlying triangle type (upper or lower) of Arg is different from the nested matrix, it will be transposed.
     * \warning This is computationally expensive unless *this and Arg are both the same triangular kind.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires internal::same_triangle_type_as<SquareRootCovariance, Arg>
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>, int> = 0>
#endif
    auto& operator*=(Arg&& arg)
    {
      if constexpr(triangular_matrix<NestedMatrix>)
      {
        nested_matrix() *= internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        // Convert both operands to triangular matrices, and then back to self-adjoint.
        const auto prod = internal::convert_nested_matrix<TBaseType>(*this) *
          internal::convert_nested_matrix<TBaseType>(std::forward<Arg>(arg));
        nested_matrix() = Cholesky_square(prod);
      }
      this->mark_changed();
      return *this;
    }


    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }

    using Base::nested_matrix;

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
    typename MatrixTraits<M>::RowCoefficients,
    nested_matrix_t<M>>;

#ifdef __cpp_concepts
  template<typed_matrix M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<nested_matrix_t<M>>::template TriangularBaseType<>>;

#ifdef __cpp_concepts
  template<covariance_nestable M>
#else
  template<typename M, std::enable_if_t<covariance_nestable<M>, int> = 0>
#endif
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<Axes<MatrixTraits<M>::dimension>, passable_t<M>>;

#ifdef __cpp_concepts
  template<typed_matrix_nestable M> requires (not covariance_nestable<M>)
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M> and not covariance_nestable<M>, int> = 0>
#endif
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
      Axes<MatrixTraits<M>::dimension>,
      typename MatrixTraits<M>::template TriangularBaseType<>>;


  // ---------------- //
  //  Make Functions  //
  // ---------------- //

  /**
   * \brief Make a SquareRootCovariance from a covariance_nestable, specifying the coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam Arg A covariance_nestable with size matching Coefficients.
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
   * \brief Make a SquareRootCovariance from a covariance_nestable, with default Axis coefficients.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam Arg A covariance_nestable.
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
   * \brief Make a SquareRootCovariance (with nested triangular matrix) from a self-adjoint typed_matrix_nestable.
   * \tparam Coefficients The coefficient types corresponding to the rows and columns.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint typed_matrix_nestable with size matching Coefficients.
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
    using T = typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>;
    return SquareRootCovariance<Coefficients, T>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a default Axis SquareRootCovariance from a self-adjoint typed_matrix_nestable.
   * \tparam TriangleType The type of the nested triangular matrix (upper, lower).
   * \tparam Arg A square, self-adjoint typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type = TriangleType::lower, typed_matrix_nestable Arg> requires
    (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<
    typed_matrix_nestable<Arg> and (not covariance_nestable<Arg>) and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_square_root_covariance<C, triangle_type>(std::forward<Arg>(arg));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, typed_matrix_nestable Arg>
    requires (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<coefficients<Coefficients> and typed_matrix_nestable<Arg> and
      (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using B = std::conditional_t<triangle_type == TriangleType::diagonal,
    typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>>;
    return SquareRootCovariance<Coefficients, B>();
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a covariance_nestable or typed_matrix_nestable.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename Arg> requires
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    constexpr TriangleType template_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularBaseType<>>;
    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<self_adjoint_matrix<Arg>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<template_type>,
        typename MatrixTraits<Arg>::template TriangularBaseType<template_type>>>;
    return SquareRootCovariance<Coefficients, B>();
  }


/**
 * \overload
 * \brief Make a writable, uninitialized SquareRootCovariance from a typed_matrix_nestable or covariance_nestable.
 * \details The coefficients will be Axis.
 */
#ifdef __cpp_concepts
  template<typename Arg> requires (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<typename Arg, std::enable_if_t<(covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
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
  template<TriangleType triangle_type, typed_matrix_nestable Arg> requires
    (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns)
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<
    typed_matrix_nestable<Arg> and (MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline auto
  make_square_root_covariance()
  {
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_square_root_covariance<C, triangle_type, Arg>();
  }


  /**
   * \overload
   * \brief Make a SquareRootCovariance based on another covariance.
   */
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_square_root_covariance<C>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance from a covariance type.
   */
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
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
   * \brief Make a SquareRootCovariance from a typed matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type = TriangleType::lower, typed_matrix Arg> requires
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>)
#else
  template<TriangleType triangle_type = TriangleType::lower, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (triangle_type == TriangleType::lower or triangle_type == TriangleType::upper) and
    (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients,
      typename MatrixTraits<Arg>::ColumnCoefficients>), int> = 0>
#endif
  inline auto
  make_square_root_covariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_square_root_covariance<C, triangle_type>(nested_matrix(std::forward<Arg>(arg)));
  }


  /**
   * \overload
   * \brief Make a writable, uninitialized SquareRootCovariance based on a typed_matrix.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg> requires
    (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>)
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients,
      typename MatrixTraits<Arg>::ColumnCoefficients>), int> = 0>
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
   * \brief Make a writable, uninitialized SquareRootCovariance based on a typed_matrix.
   */
#ifdef __cpp_concepts
  template<typed_matrix Arg> requires
    (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>)
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg> and
    (equivalent_to<typename MatrixTraits<Arg>::RowCoefficients,
      typename MatrixTraits<Arg>::ColumnCoefficients>), int> = 0>
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
    triangle_type = triangle_type_of<typename MatrixTraits<ArgType>::template TriangularBaseType<>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = SquareRootCovariance<Coeffs, self_contained_t<NestedMatrix>>;

    /// Make SquareRootCovariance from a covariance_nestable.
#ifdef __cpp_concepts
    template<coefficients C = Coeffs, covariance_nestable Arg>
#else
    template<typename C = Coeffs, typename Arg>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return SquareRootCovariance<Coeffs, NestedMatrix>::zero(); }

    static auto identity() { return SquareRootCovariance<Coeffs, NestedMatrix>::identity(); }
  };


}


#endif //OPENKALMAN_SQUAREROOTCOVARIANCE_HPP

