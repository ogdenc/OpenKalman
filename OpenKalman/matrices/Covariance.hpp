/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCE_HPP
#define OPENKALMAN_COVARIANCE_HPP

namespace OpenKalman
{
  //////////////////
  //  Covariance  //
  //////////////////

#ifdef __cpp_concepts
  template<coefficients Coeffs, covariance_nestable ArgType> requires
    (Coeffs::size == MatrixTraits<ArgType>::dimension) and (not std::is_rvalue_reference_v<ArgType>)
#else
  template<typename Coeffs, typename ArgType>
#endif
  struct Covariance : internal::CovarianceBase<Covariance<Coeffs, ArgType>, ArgType>
  {
    static_assert(covariance_nestable<ArgType>);
    static_assert(not std::is_rvalue_reference_v<ArgType>);
    using NestedMatrix = ArgType;
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    using Base = internal::CovarianceBase<Covariance, ArgType>;

  protected:
    static constexpr TriangleType storage_type =
      triangle_type_of<typename MatrixTraits<NestedMatrix>::template TriangularBaseType<>>;

    using SABaseType = std::conditional_t<diagonal_matrix<NestedMatrix>, NestedMatrix,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointBaseType<storage_type>>;

    template<typename C = Coefficients, typename Arg>
    static auto
    make(Arg&& arg) noexcept
    {
      return Covariance<C, self_contained_t<std::decay_t<Arg>>>(std::forward<Arg>(arg));
    }

  public:
    /**************
     * Constructors
     **************/

    /// Default constructor.
    Covariance() : Base() {}

    /// Copy constructor.
    Covariance(const Covariance& other) : Base(other) {}

    /// Move constructor.
    Covariance(Covariance&& other) noexcept : Base(std::move(other)) {}


    /// Convert from a general covariance type.
#ifdef __cpp_concepts
    template<covariance M> requires (not (diagonal_matrix<M> and square_root_covariance<M> and diagonal_matrix<NestedMatrix>))
#else
    template<typename M, std::enable_if_t<covariance<M> and
      not (diagonal_matrix<M> and square_root_covariance<M> and diagonal_matrix<NestedMatrix>), int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(std::forward<M>(m))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::Coefficients, Coefficients>);
    }


    /// Convert from a diagonal square-root covariance type.
#ifdef __cpp_concepts
    template<square_root_covariance M> requires diagonal_matrix<M> and diagonal_matrix<NestedMatrix>
#else
    template<typename M, std::enable_if_t<covariance<M> and
      diagonal_matrix<M> and square_root_covariance<M> and diagonal_matrix<NestedMatrix>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(Cholesky_square(std::forward<M>(m).nested_matrix()))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::Coefficients, Coefficients>);
    }


    /// Construct from a covariance_nestable, general case.
#ifdef __cpp_concepts
    template<covariance_nestable M> requires
      (diagonal_matrix<M> or diagonal_matrix<NestedMatrix> or
        (self_adjoint_matrix<M> and self_adjoint_matrix<NestedMatrix>) or
        internal::same_triangle_type_as<M, NestedMatrix>) and
      (not diagonal_matrix<M> or self_adjoint_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      (diagonal_matrix<M> or diagonal_matrix<NestedMatrix> or
        (self_adjoint_matrix<M> and self_adjoint_matrix<NestedMatrix>) or
        internal::same_triangle_type_as<M, NestedMatrix>) and
      (not diagonal_matrix<M> or self_adjoint_matrix<NestedMatrix>), int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(std::forward<M>(m)) {}


    /// Construct from a covariance_nestable, if it has a different triangle type.
#ifdef __cpp_concepts
    template<covariance_nestable M> requires (not diagonal_matrix<M>) and
      (not diagonal_matrix<NestedMatrix>) and
      (not (self_adjoint_matrix<M> and self_adjoint_matrix<NestedMatrix>)) and
      (not internal::same_triangle_type_as<M, NestedMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      not diagonal_matrix<M> and not diagonal_matrix<NestedMatrix> and
      (not (self_adjoint_matrix<M> and self_adjoint_matrix<NestedMatrix>)) and
        (not internal::same_triangle_type_as<M, NestedMatrix>), int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(adjoint(std::forward<M>(m))) {}


    /// Construct from a covariance_nestable, diagonal case that needs to be squared.
#ifdef __cpp_concepts
    template<covariance_nestable M> requires diagonal_matrix<M> and (not self_adjoint_matrix<NestedMatrix>)
#else
    template<typename M, std::enable_if_t<covariance_nestable<M> and
      diagonal_matrix<M> and not self_adjoint_matrix<NestedMatrix>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(Cholesky_factor(std::forward<M>(m))) {}


    /// Construct from a typed matrix (assumed to be self-adjoint).
#ifdef __cpp_concepts
    template<typed_matrix M>
#else
    template<typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(MatrixTraits<SABaseType>::make(OpenKalman::nested_matrix(std::forward<M>(m))))
    {
      static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      if constexpr(not diagonal_matrix<NestedMatrix>)
        static_assert(equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, Coefficients>);
    }

    /// Construct from a typed_matrix_nestable (assumed to be self-adjoint).
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires (not covariance_nestable<M>)
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and not covariance_nestable<M>, int> = 0>
#endif
    Covariance(M&& m) noexcept : Base(MatrixTraits<SABaseType>::make(std::forward<M>(m))) {}

    /// Construct from Scalar coefficients. Assumes matrix is self-adjoint, and only reads lower left triangle.
    template<typename ... Args, std::enable_if_t< std::conjunction_v<std::is_convertible<Args, const Scalar>...>, int> = 0>
    Covariance(Args ... args) : Base(MatrixTraits<SABaseType>::make(args...)) {}

    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    auto& operator=(const Covariance& other)
    {
      if constexpr(not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(other);
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(Covariance&& other) noexcept
    {
      if constexpr(not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>)
        Base::operator=(std::move(other));
      return *this;
    }

    /// Assign from a compatible covariance type.
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

      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else if constexpr(covariance<Arg> and
        diagonal_matrix<Arg> and square_root_covariance<Arg> and diagonal_matrix<NestedMatrix>)
      {
        Base::operator=(Cholesky_square(std::forward<Arg>(other).nested_matrix()));
      }
      else
      {
        Base::operator=(internal::convert_nested_matrix<std::decay_t<NestedMatrix>>(std::forward<Arg>(other)));
      }
      return *this;
    }

#ifdef __cpp_concepts
    template<covariance Arg>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
    auto& operator+=(Arg&& arg) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(not square_root_covariance<Arg>);
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        this->nested_matrix() += internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        decltype(auto) E1 = this->nested_matrix();
        decltype(auto) E2 = internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg));
        if constexpr(upper_triangular_matrix<NestedMatrix>)
          this->nested_matrix() = QR_decomposition(concatenate_vertical(E1, E2));
        else
          this->nested_matrix() = LQ_decomposition(concatenate_horizontal(E1, E2));
      }
      this->mark_changed();
      return *this;
    }

    auto& operator+=(const Covariance& arg) noexcept
    {
      return operator+=<const Covariance&>(arg);
    }


#ifdef __cpp_concepts
    template<covariance Arg>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
    auto& operator-=(Arg&& arg) noexcept
    {
      static_assert(equivalent_to<typename MatrixTraits<Arg>::Coefficients, Coefficients>);
      static_assert(not square_root_covariance<Arg>);
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        this->nested_matrix() -= internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg));
      }
      else
      {
        using TLowerType = typename MatrixTraits<NestedMatrix>::template TriangularBaseType<TriangleType::lower>;
        const auto U = internal::convert_nested_matrix<TLowerType>(std::forward<Arg>(arg));
        rank_update(this->nested_matrix(), U, Scalar(-1));
      }
      this->mark_changed();
      return *this;
    }

    auto& operator-=(const Covariance& arg) noexcept
    {
      return operator-=<const Covariance&>(arg);
    }

#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto& operator*=(const S s)
    {
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        this->nested_matrix() *= s;
      }
      else
      {
        if (s > S(0))
        {
          this->nested_matrix() *= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < S(0))
        {
          using TLowerType = typename MatrixTraits<NestedMatrix>::template TriangularBaseType<TriangleType::lower>;
          const auto U = internal::convert_nested_matrix<TLowerType>(*this);
          this->nested_matrix() = MatrixTraits<NestedMatrix>::zero();
          rank_update(this->nested_matrix(), U, s);
        }
        else
        {
          this->nested_matrix() = MatrixTraits<NestedMatrix>::zero();
        }
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
      if constexpr(self_adjoint_matrix<NestedMatrix>)
      {
        this->nested_matrix() /= s;
      }
      else
      {
        if (s > S(0))
        {
          this->nested_matrix() /= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < S(0))
        {
          using TLowerType = typename MatrixTraits<NestedMatrix>::template TriangularBaseType<TriangleType::lower>;
          const auto u = internal::convert_nested_matrix<TLowerType>(*this);
          this->nested_matrix() = MatrixTraits<NestedMatrix>::zero();
          rank_update(this->nested_matrix(), u, 1 / static_cast<Scalar>(s));
        }
        else
        {
          throw (std::runtime_error("Covariance operator/=: divide by zero"));
        }
      }
      this->mark_changed();
      return *this;
    }

    /// Scale by a factor. Equivalent to multiplication by the square of a scalar.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto&
    scale(const S s)
    {
      if constexpr(self_adjoint_matrix<NestedMatrix>)
        this->nested_matrix() *= static_cast<Scalar>(s) * s;
      else
        this->nested_matrix() *= s;
      this->mark_changed();
      return *this;
    }

    /// Scale by the inverse of a scalar factor. Equivalent by division by the square of a scalar.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> S>
#else
    template<typename S, std::enable_if_t<std::is_convertible_v<S, Scalar>, int> = 0>
#endif
    auto&
    inverse_scale(const S s)
    {
      if constexpr(self_adjoint_matrix<NestedMatrix>)
        this->nested_matrix() /= static_cast<Scalar>(s) * s;
      else
        this->nested_matrix() /= s;
      this->mark_changed();
      return *this;
    }


    /*********
     * Other
     *********/

    static auto zero() { return make(MatrixTraits<NestedMatrix>::zero()); }

    static auto identity() { return make(MatrixTraits<NestedMatrix>::identity()); }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  Covariance(M&&) -> Covariance<typename MatrixTraits<M>::Coefficients, nested_matrix_t<M>>;


#ifdef __cpp_concepts
  template<covariance_nestable M>
#else
  template<typename M, std::enable_if_t<covariance_nestable<M>, int> = 0>
#endif
  Covariance(M&&) -> Covariance<Axes<MatrixTraits<M>::dimension>, passable_t<M>>;


#ifdef __cpp_concepts
  template<typed_matrix M>
#else
  template<typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
  Covariance(M&&) -> Covariance<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<nested_matrix_t<M>>::template SelfAdjointBaseType<>>;

#ifdef __cpp_concepts
  template<typed_matrix_nestable M> requires (not covariance_nestable<M>)
#else
  template<typename M, std::enable_if_t<typed_matrix_nestable<M> and not covariance_nestable<M>, int> = 0>
#endif
  Covariance(M&&) -> Covariance<
    Axes<MatrixTraits<M>::dimension>, typename MatrixTraits<M>::template SelfAdjointBaseType<>>;


  //////////////////////
  //  Make Functions  //
  //////////////////////

  /// Make a Covariance based on a covariance_nestable.
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    return Covariance<Coefficients, passable_t<Arg>>(std::forward<Arg>(arg));
  }


  /// Make a Covariance, converting from a matrix other than a covariance_nestable.
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType ... triangle_type, typed_matrix_nestable Arg> requires
    (sizeof...(triangle_type) <= 1) and (not covariance_nestable<Arg>)
#else
  template<typename Coefficients, TriangleType ... triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and not covariance_nestable<Arg> and
      typed_matrix_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType arg_t_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularBaseType<>>;
    constexpr TriangleType t_type = std::get<0>(std::tuple {triangle_type..., arg_t_type});
    using T = typename MatrixTraits<Arg>::template TriangularBaseType<t_type>;
    using SA = typename MatrixTraits<Arg>::template SelfAdjointBaseType<t_type>;
    using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
    SA b = std::forward<Arg>(arg);
    return Covariance<Coefficients, B> {b};
  }


  /// Make an axes-only Covariance, based on a covariance_nestable or regular matrix.
#ifdef __cpp_concepts
  template<TriangleType ... triangle_type, typename Arg> requires
    (sizeof...(triangle_type) == 0 and covariance_nestable<Arg>) or
    (sizeof...(triangle_type) <= 1 and typed_matrix_nestable<Arg>)
#else
  template<TriangleType ... triangle_type, typename Arg, std::enable_if_t<
    ((sizeof...(triangle_type)) == 0 and covariance_nestable<Arg>) or
    ((sizeof...(triangle_type)) <= 1 and typed_matrix_nestable<Arg>), int> = 0>
#endif
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_Covariance<C, triangle_type...>(std::forward<Arg>(arg));
  }


  /// Make a default Covariance, based on a template type, for a covariance_nestable or regular matrix.
#ifdef __cpp_concepts
  template<typename Coefficients, typename Arg> requires covariance_nestable<Arg> or typed_matrix_nestable<Arg>
#else
  template<typename Coefficients, typename Arg,
    std::enable_if_t<covariance_nestable<Arg> or typed_matrix_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    constexpr TriangleType triangle_type = triangle_type_of<typename MatrixTraits<Arg>::template TriangularBaseType<>>;

    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      std::conditional_t<triangular_matrix<Arg>,
        typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>,
        typename MatrixTraits<Arg>::template SelfAdjointBaseType<triangle_type>>>;

    return Covariance<Coefficients, B>();
  }


  /// Make a default Covariance, based on a template type, for a regular matrix.
#ifdef __cpp_concepts
  template<coefficients Coefficients, TriangleType triangle_type, typed_matrix_nestable Arg>
#else
  template<typename Coefficients, TriangleType triangle_type, typename Arg,
    std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);

    using B = std::conditional_t<diagonal_matrix<Arg>,
      typename MatrixTraits<Arg>::template DiagonalBaseType<>,
      typename MatrixTraits<Arg>::template TriangularBaseType<triangle_type>>;

    return Covariance<Coefficients, B>();
  }


  /// Make a default axes-only Covariance, based on a template type.
#ifdef __cpp_concepts
  template<typename Arg> requires covariance_nestable<Arg> or typed_matrix_nestable<Arg>
#else
  template<typename Arg, std::enable_if_t<covariance_nestable<Arg> or typed_matrix_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_Covariance<C, Arg>();
  }


  /// Make a default axes-only Covariance for a regular matrix.
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix_nestable Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix_nestable<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    using C = Axes<MatrixTraits<Arg>::dimension>;
    return make_Covariance<C, triangle_type, Arg>();
  }


  // Make from another covariance type

  /// Make a Covariance based on another covariance.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    return make_Covariance<C>(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default Covariance, based on a covariance template type.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    using C = typename MatrixTraits<Arg>::Coefficients;
    using B = nested_matrix_t<Arg>;
    return make_Covariance<C, B>();
  }


  // Make from a typed matrix

  /// Make a Covariance from a typed matrix.
#ifdef __cpp_concepts
  template<TriangleType...triangle_type, typed_matrix Arg> requires (sizeof...(triangle_type) <= 1)
#else
  template<TriangleType...triangle_type, typename Arg,
    std::enable_if_t<sizeof...(triangle_type) <= 1 and typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance(Arg&& arg) noexcept
  {
    static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    if constexpr(covariance_nestable<nested_matrix_t<Arg>>)
      return make_Covariance<C>(nested_matrix(std::forward<Arg>(arg)));
    else
      return make_Covariance<C, triangle_type...>(nested_matrix(std::forward<Arg>(arg)));
  }


  /// Make a default axes-only Covariance, based on a typed matrix template type, specifying a triangle type.
#ifdef __cpp_concepts
  template<TriangleType triangle_type, typed_matrix Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
    return make_Covariance<C, triangle_type, B>();
  }


  /// Make a default axes-only Covariance, based on a typed matrix template type.
#ifdef __cpp_concepts
  template<typed_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<typed_matrix<Arg>, int> = 0>
#endif
  inline auto
  make_Covariance()
  {
    static_assert(equivalent_to<typename MatrixTraits<Arg>::RowCoefficients, typename MatrixTraits<Arg>::ColumnCoefficients>);
    using C = typename MatrixTraits<Arg>::RowCoefficients;
    using B = nested_matrix_t<Arg>;
    return make_Covariance<C, B>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename ArgType>
  struct MatrixTraits<Covariance<Coeffs, ArgType>>
  {
    using NestedMatrix = ArgType;
    static constexpr auto dimension = MatrixTraits<NestedMatrix>::dimension;
    static constexpr auto columns = dimension;
    static_assert(Coeffs::size == dimension);
    using Coefficients = Coeffs;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar; ///< Scalar type for this vector.

    template<std::size_t rows = dimension, std::size_t cols = dimension, typename S = Scalar>
    using NativeMatrix = typename MatrixTraits<NestedMatrix>::template NativeMatrix<rows, cols, S>;

    using SelfContained = Covariance<Coefficients, self_contained_t<NestedMatrix>>;

    /// Make covariance from a covariance_nestable.
#ifdef __cpp_concepts
    template<coefficients C = Coefficients, covariance_nestable Arg>
#else
    template<typename C = Coefficients, typename Arg>
#endif
    static auto make(Arg&& arg) noexcept
    {
      return Covariance<C, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }

    static auto zero() { return Covariance<Coefficients, NestedMatrix>::zero(); }

    static auto identity() { return Covariance<Coefficients, NestedMatrix>::identity(); }
  };


} // OpenKalman

#endif //OPENKALMAN_COVARIANCE_HPP
