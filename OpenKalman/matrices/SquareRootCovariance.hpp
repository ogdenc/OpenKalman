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
  template<coefficients Coeffs, covariance_base ArgType> requires
    (Coeffs::size == MatrixTraits<ArgType>::dimension) and (not std::is_rvalue_reference_v<ArgType>)
#else
  template<typename Coeffs, typename ArgType>
#endif
  struct SquareRootCovariance : internal::CovarianceBase<SquareRootCovariance<Coeffs, ArgType>, ArgType>
  {
    static_assert(Coeffs::size == MatrixTraits<ArgType>::dimension);
    static_assert(not std::is_rvalue_reference_v<ArgType>);
    using BaseMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    using Base = internal::CovarianceBase<SquareRootCovariance, ArgType>;
    static constexpr TriangleType triangle_type =
      triangle_type_of<typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>;

  protected:
    using TBaseType = std::conditional_t<diagonal_matrix<BaseMatrix>, BaseMatrix,
      typename MatrixTraits<BaseMatrix>::template TriangularBaseType<triangle_type>>;

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
      (not (diagonal_matrix<M> and not square_root_covariance<M> and diagonal_matrix<BaseMatrix>))
#else
    template<typename M, std::enable_if_t<covariance<M> and
      (not (diagonal_matrix<M> and not square_root_covariance<M> and diagonal_matrix<BaseMatrix>)), int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(std::forward<M>(m))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::Coefficients, Coefficients>);
      using MBase = nested_matrix_t<M>;
      static_assert(not square_root_covariance<M> or self_adjoint_matrix<MBase> or self_adjoint_matrix<BaseMatrix> or
          internal::same_triangle_type_as<BaseMatrix, MBase>,
        "An upper-triangle Cholesky-form covariance cannot be constructed from a lower-triangle Cholesky-form "
        "covariance, and vice versa. To convert, use adjoint().");
    }


    /// Construct from a general covariance type.
#ifdef __cpp_concepts
    template<covariance M> requires
      diagonal_matrix<M> and (not square_root_covariance<M>) and diagonal_matrix<BaseMatrix>
#else
    template<typename M, std::enable_if_t<covariance<M> and
      diagonal_matrix<M> and not square_root_covariance<M> and diagonal_matrix<BaseMatrix>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(Cholesky_factor(std::forward<M>(m).base_matrix()))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::Coefficients, Coefficients>);
    }


    /// Construct from a non-diagonal covariance base.
#ifdef __cpp_concepts
    template<covariance_base M> requires (not diagonal_matrix<M> or triangular_matrix<BaseMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_base<M> and
      (not diagonal_matrix<M> or triangular_matrix<BaseMatrix>), int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(std::forward<M>(m)) {}


    /// Construct from a diagonal covariance base.
#ifdef __cpp_concepts
    template<covariance_base M> requires diagonal_matrix<M> and (not triangular_matrix<BaseMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_base<M> and
      diagonal_matrix<M> and not triangular_matrix<BaseMatrix>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept : Base(Cholesky_square(std::forward<M>(m))) {}


    /// Construct from a typed matrix (assumed to be triangular).
#ifdef __cpp_concepts
    template<typed_matrix M>
#else
    template<typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
    SquareRootCovariance(M&& m) noexcept
      : Base(MatrixTraits<TBaseType>::make(OpenKalman::base_matrix(std::forward<M>(m))))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      if constexpr(diagonal_matrix<BaseMatrix>)
        static_assert(MatrixTraits<M>::columns == 1);
      else
        static_assert(equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, Coefficients>);
    }


    /// Construct from a typed matrix base (assumed to be triangular).
#ifdef __cpp_concepts
    template<typed_matrix_base M> requires (not covariance_base<M>)
#else
    template<typename M, std::enable_if_t<typed_matrix_base<M> and not covariance_base<M>, int> = 0>
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
      if constexpr(not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>)
        Base::operator=(other);
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(SquareRootCovariance&& other) noexcept
    {
      if constexpr(not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>)
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
        static_assert(equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      }
      else if constexpr(typed_matrix<Arg>)
      {
        static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients> and
          equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, Coefficients>);
      }
      using ArgBase = nested_matrix_t<Arg>;
      static_assert(not square_root_covariance<Arg> or self_adjoint_matrix<ArgBase> or self_adjoint_matrix<BaseMatrix> or
          internal::same_triangle_type_as<BaseMatrix, ArgBase>,
          "An upper-triangle Cholesky-form covariance cannot be assigned a lower-triangle Cholesky-form "
          "covariance, and vice versa. To convert, use adjoint().");

      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(covariance<Arg> and
        diagonal_matrix<Arg> and not square_root_covariance<Arg> and diagonal_matrix<BaseMatrix>)
      {
        Base::operator=(Cholesky_factor(std::forward<Arg>(other).base_matrix()));
      }
      else
      {
        Base::operator=(internal::convert_base_matrix<std::decay_t<BaseMatrix>>(std::forward<Arg>(other)));
      }
      return *this;
    }


    /**
     * Increment by another square-root (Cholesky) covariance.
     * \warning This is computationally expensive if the base matrix is self-adjoint. This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& arg) noexcept
    {
      if constexpr(triangular_matrix<BaseMatrix>)
      {
        base_matrix() += internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        const auto sum = internal::convert_base_matrix<TBaseType>(*this) +
          internal::convert_base_matrix<TBaseType>(std::forward<Arg>(arg));
        base_matrix() = Cholesky_square(sum);
      }
      this->mark_changed();
      return *this;
    }


    /**
     * Increment by another square-root (Cholesky) covariance of the same type.
     * \warning This is computationally expensive if the base matrix is self-adjoint. This can generally be avoided.
     */
    auto& operator+=(const SquareRootCovariance& arg)
    {
      return operator+=<const SquareRootCovariance&>(arg);
    }


    /**
     * Decrement by another square-root (Cholesky) covariance.
     * \warning This is computationally expensive if the base matrix is self-adjoint. This can generally be avoided.
     */
#ifdef __cpp_concepts
    template<square_root_covariance Arg> requires
      equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>
#else
    template<typename Arg, std::enable_if_t<square_root_covariance<Arg> and
      equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients> and
      internal::same_triangle_type_as<SquareRootCovariance, Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& arg) noexcept
    {
      if constexpr(triangular_matrix<BaseMatrix>)
      {
        base_matrix() -= internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        const auto diff = internal::convert_base_matrix<TBaseType>(*this) -
          internal::convert_base_matrix<TBaseType>(std::forward<Arg>(arg));
        base_matrix() = Cholesky_square(diff);
      }
      this->mark_changed();
      return *this;
    }

    /**
     * Decrement by another square-root (Cholesky) covariance of the same type.
     * \warning This is computationally expensive if the base matrix is self-adjoint. This can generally be avoided.
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
      if constexpr(triangular_matrix<BaseMatrix>)
      {
        base_matrix() *= s;
      }
      else
      {
        base_matrix() *= static_cast<Scalar>(s) * s;
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
      if constexpr(triangular_matrix<BaseMatrix>)
      {
        base_matrix() /= s;
      }
      else
      {
        base_matrix() /= static_cast<Scalar>(s) * s;
      }
      this->mark_changed();
      return *this;
    }


    /**
     * \brief Multiply by another square-root covariance matrix.
     * \details If the underlying triangle type (upper or lower) of Arg is different from the base matrix, it will be transposed.
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
      if constexpr(triangular_matrix<BaseMatrix>)
      {
        base_matrix() *= internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        // Convert both operands to triangular matrices, and then back to self-adjoint.
        const auto prod = internal::convert_base_matrix<TBaseType>(*this) *
          internal::convert_base_matrix<TBaseType>(std::forward<Arg>(arg));
        base_matrix() = Cholesky_square(prod);
      }
      this->mark_changed();
      return *this;
    }


    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<BaseMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<BaseMatrix>::identity()); }

    using Base::base_matrix;

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
    typename MatrixTraits<M>::Coefficients,
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
  template<covariance_base M>
#else
  template<typename M, std::enable_if_t<covariance_base<M>, int> = 0>
#endif
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<Axes<MatrixTraits<M>::dimension>, passable_t<M>>;

#ifdef __cpp_concepts
  template<typed_matrix_base M> requires (not covariance_base<M>)
#else
  template<typename M, std::enable_if_t<typed_matrix_base<M> and not covariance_base<M>, int> = 0>
#endif
  SquareRootCovariance(M&&)
    -> SquareRootCovariance<
      Axes<MatrixTraits<M>::dimension>,
      typename MatrixTraits<M>::template TriangularBaseType<>>;


  //////////////////////
  //  Make Functions  //
  //////////////////////

  // Make from covariance base or regular matrix:

  /// Make a SquareRootCovariance based on a covariance base.
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_base Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<covariance_base<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance(Arg&& arg) noexcept
  {
    return SquareRootCovariance<Coefficients, passable_t<Arg>>(std::forward<Arg>(arg));
  }


  /// Make a SquareRootCovariance, converting from a matrix other than a covariance base.
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType...triangle_type, typed_matrix_base Arg> requires
    (sizeof...(triangle_type) <= 1) and (not covariance_base<Arg>)
#else
  template<typename Coefficients, TriangleType...triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and not covariance_base<Arg> and
      typed_matrix_base<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType arg_t_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularBaseType<>>;
    constexpr TriangleType t_type = std::get<0>(std::tuple {triangle_type..., arg_t_type});
    using B = typename MatrixTraits<Arg>::template TriangularBaseType<t_type>;
    return SquareRootCovariance<Coefficients, B> {static_cast<B>(std::forward<Arg>(arg))};
  }


  /// Make an axes-only SquareRootCovariance, based on a covariance base or regular matrix.
#ifdef __cpp_concepts
  template<TriangleType...triangle_type, typename Arg> requires
    (sizeof...(triangle_type) == 0 and covariance_base<Arg>) or
    (sizeof...(triangle_type) <= 1 and typed_matrix_base<Arg>)
#else
  template<TriangleType...triangle_type, typename Arg, std::enable_if_t<
    (sizeof...(triangle_type) == 0 and covariance_base<Arg>) or
    (sizeof...(triangle_type) <= 1 and typed_matrix_base<Arg>), int> = 0>
#endif
  inline auto
  make_SquareRootCovariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_SquareRootCovariance<C, triangle_type...>(std::forward<Arg>(arg));
  }


  /// Make a default SquareRootCovariance, based on a template type.
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename Arg> requires covariance_base<Arg> or typed_matrix_base<Arg>
#else
  template<typename Coefficients, typename Arg,
    std::enable_if_t<covariance_base<Arg> or typed_matrix_base<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType template_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularBaseType<>>;

    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<self_adjoint_matrix<Arg>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<template_type>,
        typename MatrixTraits<Arg>::template TriangularBaseType<template_type>>>;

    return SquareRootCovariance<Coefficients, B>();
  }


  /// Make a default SquareRootCovariance for a regular matrix.
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, typed_matrix_base Arg>
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<typed_matrix_base<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);

    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<self_adjoint_matrix<Arg>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<triangle_type>,
        typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>>>;

    return SquareRootCovariance<Coefficients, B>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a template type.
#ifdef __cpp_concepts
  template<typename Arg> requires covariance_base<Arg> or typed_matrix_base<Arg>
#else
  template<typename Arg, std::enable_if_t<covariance_base<Arg> or typed_matrix_base<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_SquareRootCovariance<C, Arg>();
  }


  /// Make a default axes-only SquareRootCovariance for a regular matrix.
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix_base Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix_base<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_SquareRootCovariance<C, triangle_type, Arg>();
  }


  // Make from another covariance type

  /// Make a SquareRootCovariance based on another covariance.
#ifdef __cpp_concepts
  template<TriangleType...triangle_type, covariance Arg> requires (sizeof...(triangle_type) <= 1)
#else
  template<TriangleType...triangle_type, typename Arg, std::enable_if_t<
    sizeof...(triangle_type) <= 1 and covariance<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_SquareRootCovariance<C, triangle_type...>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default axes-only SquareRootCovariance, based on a covariance template type.
#ifdef __cpp_concepts
  template<TriangleType triangle_type, covariance Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    using B = nested_matrix_t<Arg>;
    return make_SquareRootCovariance<C, triangle_type, B>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a covariance template type.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    using B = nested_matrix_t<Arg>;
    return make_SquareRootCovariance<C, B>();
  }


  // Make from a typed matrix

  /// Make a SquareRootCovariance from a typed matrix.
#ifdef __cpp_concepts
  template<TriangleType...triangle_type, typed_matrix Arg> requires (sizeof...(triangle_type) <= 1)
#else
  template<TriangleType...triangle_type, typename Arg, std::enable_if_t<
    sizeof...(triangle_type) <= 1 and typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance(Arg&& arg) noexcept
  {
    static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    return make_SquareRootCovariance<C, triangle_type...>(base_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default axes-only SquareRootCovariance, based on a typed matrix template type, specifying a triangle type.
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
    return make_SquareRootCovariance<C, triangle_type, B>();
  }


  /// Make a default axes-only SquareRootCovariance, based on a typed matrix template type.
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_SquareRootCovariance()
  {
    static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
    return make_SquareRootCovariance<C, B>();
  }


  ////////////////////////////
  //        Traits          //
  ////////////////////////////

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<SquareRootCovariance<Coeffs, ArgType>>
  {
    using BaseMatrix = ArgType;
    static constexpr auto dimension = MatrixTraits<BaseMatrix>::dimension;
    static constexpr auto columns = dimension;
    static_assert(Coeffs::size == dimension);
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar; ///< Scalar type for this vector.

    static constexpr TriangleType
    triangle_type = triangle_type_of<typename MatrixTraits<ArgType>::template TriangularBaseType<>>;

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<BaseMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = SquareRootCovariance<Coefficients, self_contained_t<BaseMatrix>>;

    /// Make SquareRootCovariance from a covariance base.
#ifdef __cpp_concepts
    template<coefficients C = Coefficients, covariance_base Arg>
#else
    template<typename C = Coefficients, typename Arg>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return SquareRootCovariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return SquareRootCovariance<Coefficients, BaseMatrix>::zero(); }

    static auto identity() { return SquareRootCovariance<Coefficients, BaseMatrix>::identity(); }
  };


}


#endif //OPENKALMAN_SQUAREROOTCOVARIANCE_HPP

