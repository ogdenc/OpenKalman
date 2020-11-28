/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEBASE_H
#define OPENKALMAN_COVARIANCEBASE_H

#include <utility>

namespace OpenKalman::internal
{
  /**
   * \internal
   * Base of Covariance and SquareRootCovariance classes, if ArgType is not an lvalue reference and either
   * (1) Derived is not a square root and the nested matrix is self-adjoint; or
   * (2) Derived is a square root and the nested matrix is triangular.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    ((not square_root_covariance<Derived> and self_adjoint_matrix<ArgType>) or
      (square_root_covariance<Derived> and triangular_matrix<ArgType>)) and
    (not std::is_lvalue_reference_v<ArgType>) and
    (not internal::contains_nested_lvalue_reference<ArgType>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    ((not square_root_covariance<Derived> and self_adjoint_matrix<ArgType>) or
      (square_root_covariance<Derived> and triangular_matrix<ArgType>)) and
    (not std::is_lvalue_reference_v<ArgType>) and
    (not internal::contains_nested_lvalue_reference<ArgType>)>>
#endif
    : CovarianceBaseBase<Derived, ArgType>
  {
    using NestedMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, NestedMatrix>;
    using Base::Base;
    using Base::operator=;
    using Base::nested_matrix;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;


    auto operator() (std::size_t i, std::size_t j)
    {
      return make_ElementSetter<not element_settable<Derived, 2>>(nested_matrix(), i, j);
    }


    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(nested_matrix(), i, j);
    }


    auto operator[] (std::size_t i)
    {
      return make_ElementSetter<not element_settable<Derived, 2>>(nested_matrix(), i);
    }


    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(nested_matrix(), i);
    }


    auto operator() (std::size_t i) { return operator[](i); }


    auto operator() (std::size_t i) const { return operator[](i); }


  protected:
    template<typename T, typename Arg>
#ifdef __cpp_concepts
    requires (std::is_void_v<T> or covariance_nestable<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
    friend constexpr decltype(auto) convert_nested_matrix(Arg&&) noexcept;


    constexpr void mark_changed() const {}


    /// Get the apparent nested matrix.
    constexpr auto& get_apparent_nested_matrix() & { return nested_matrix(); }


    /// Get the apparent nested matrix.
    constexpr auto&& get_apparent_nested_matrix() && { return std::move(nested_matrix()); }


    /// Get the apparent nested matrix.
    constexpr const auto& get_apparent_nested_matrix() const & { return nested_matrix(); }


    /// Get the apparent nested matrix.
    constexpr const auto&& get_apparent_nested_matrix() const && { return std::move(nested_matrix()); }

  };


  // ============================================================================
  /**
   * \internal
   * Base of Covariance and SquareRootCovariance classes, if ArgType is an lvalue reference.
   * No conversion is necessary if either
   * (1) Derived is not a square root and the nested matrix is self-adjoint; or
   * (2) Derived is a square root and the nested matrix is triangular.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    ((not square_root_covariance<Derived> and self_adjoint_matrix<ArgType>) or
      (square_root_covariance<Derived> and triangular_matrix<ArgType>)) and
    (std::is_lvalue_reference_v<ArgType> or internal::contains_nested_lvalue_reference<ArgType>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    ((not square_root_covariance<Derived> and self_adjoint_matrix<ArgType>) or
      (square_root_covariance<Derived> and triangular_matrix<ArgType>)) and
    (std::is_lvalue_reference_v<ArgType> or internal::contains_nested_lvalue_reference<ArgType>)>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using NestedMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, NestedMatrix>;
    using Base::nested_matrix;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  private:
    const bool apparent_nested_linked;
    bool* const synchronized;

  public:
    /// Default constructor.
    CovarianceBase() = delete;


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base(other.nested_matrix()), synchronized(other.synchronized) {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base(std::move(other)), synchronized(other.synchronized) {}


    /// Construct from a covariance_nestable or another covariance that does not store a distinct apparent nested matrix.
#ifdef __cpp_concepts
    template<typename Arg> requires covariance_nestable<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>)))
#else
    template<typename Arg,
      std::enable_if_t<covariance_nestable<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>))), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)), apparent_nested_linked(false), synchronized(new bool {true}) {}


    /// Construct from another covariance that stores a distinct apparent nested matrix (nested matrix is not an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>))) and
      (not (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>))
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
      ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>)) and not
      (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)), apparent_nested_linked(true), synchronized(&arg.synchronized) {}


    /// Construct from another covariance that stores a distinct apparent nested matrix (nested matrix is an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>))) and
      (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
      ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>)) and
      (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)), apparent_nested_linked(true), synchronized(arg.synchronized) {}


    ~CovarianceBase()
    {
      if (not apparent_nested_linked)
      {
        delete synchronized;
      }
    }


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        Base::operator=(other);
        if (apparent_nested_linked) *synchronized = false;
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        Base::operator=(std::move(other));
        if (apparent_nested_linked) *synchronized = false;
      }
      return *this;
    }


    /// Assign from a covariance_nestable or typed_matrix_nestable.
#ifdef __cpp_concepts
    template<typename Arg> requires (covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
      std::is_assignable_v<NestedMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(covariance_nestable<Arg> or typed_matrix_nestable<Arg>) and
      std::is_assignable_v<NestedMatrix, Arg&&>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        Base::operator=(std::forward<Arg>(arg));
        if (apparent_nested_linked) *synchronized = false;
      }
      return *this;
    }


    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr (element_settable<Derived, 2>)
        return ElementSetter(nested_matrix(), i, j, [] {}, [this] { if (apparent_nested_linked) *synchronized = false; });
      else
        return make_ElementSetter<true>(nested_matrix(), i, j);
    }


    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(nested_matrix(), i, j);
    }


    auto operator[] (std::size_t i)
    {
      if constexpr (element_settable<Derived, 1>)
        return ElementSetter(nested_matrix(), i, [] {}, [this] { if (apparent_nested_linked) *synchronized = false; });
      else
        return make_ElementSetter<true>(nested_matrix(), i);
    }


    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(nested_matrix(), i);
    }


    auto operator() (std::size_t i) { return operator[](i); }


    auto operator() (std::size_t i) const { return operator[](i); }


  protected:
    template<typename T, typename Arg>
#ifdef __cpp_concepts
    requires (std::is_void_v<T> or covariance_nestable<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
    friend constexpr decltype(auto) convert_nested_matrix(Arg&&) noexcept;


    constexpr void mark_changed() const {}


    /// Get the apparent nested matrix.
    constexpr auto& get_apparent_nested_matrix() & { return nested_matrix(); }


    /// Get the apparent nested matrix.
    constexpr auto&& get_apparent_nested_matrix() && { return std::move(nested_matrix()); }


    /// Get the apparent nested matrix.
    constexpr const auto& get_apparent_nested_matrix() const & { return nested_matrix(); }


    /// Get the apparent nested matrix.
    constexpr const auto&& get_apparent_nested_matrix() const && { return std::move(nested_matrix()); }

  };


  // ============================================================================
  /**
   * \internal
   * Base of Covariance and SquareRootCovariance classes, if ArgType is not an lvalue reference, and
   * # Derived is a square root and the nested matrix is not triangular (i.e., it is self-adjoint but not diagonal); or
   * # Derived is not a square root and the nested matrix is not self-adjoint (i.e., it is triangular but not diagonal).
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    (square_root_covariance<Derived> or not self_adjoint_matrix<ArgType>) and
    (not square_root_covariance<Derived> or not triangular_matrix<ArgType>) and
    (not std::is_lvalue_reference_v<ArgType>) and
    (not internal::contains_nested_lvalue_reference<ArgType>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    (square_root_covariance<Derived> or not self_adjoint_matrix<ArgType>) and
    (not square_root_covariance<Derived> or not triangular_matrix<ArgType>) and
    (not std::is_lvalue_reference_v<ArgType>) and
    (not internal::contains_nested_lvalue_reference<ArgType>)>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using NestedMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, NestedMatrix>;
    using Base::nested_matrix;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  private:
    using ApparentNestedMatrix = std::conditional_t<triangular_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointBaseType<>,
      typename MatrixTraits<NestedMatrix>::template TriangularBaseType<>>;

    mutable bool synchronized;

    mutable ApparentNestedMatrix apparent_nested_matrix; ///< The apparent nested matrix for Covariance or SquareRootCovariance.

    void synchronize() const
    {
      if constexpr(square_root_covariance<Derived>)
        apparent_nested_matrix = Cholesky_factor(nested_matrix());
      else
        apparent_nested_matrix = Cholesky_square(nested_matrix());
      synchronized = true;
    }

  public:
    /// Default constructor.
    CovarianceBase() : synchronized() {}

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base(other),
        synchronized(other.synchronized),
        apparent_nested_matrix(other.apparent_nested_matrix) {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base(std::move(other.nested_matrix())),
        synchronized(other.synchronized),
        apparent_nested_matrix(std::move(other.apparent_nested_matrix)) {}


    /// Construct from a general covariance type. Argument matches apparent nested matrix.
#ifdef __cpp_concepts
    template<covariance Arg> requires
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) ==
        square_root_covariance<Derived>) and
      internal::same_triangle_type_as<nested_matrix_t<Arg>, ApparentNestedMatrix>
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) ==
        square_root_covariance<Derived>) and
      internal::same_triangle_type_as<nested_matrix_t<Arg>, ApparentNestedMatrix>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(internal::convert_nested_matrix<NestedMatrix>(arg)),
        synchronized(true),
        apparent_nested_matrix(std::forward<Arg>(arg).nested_matrix()) {}


    /// Construct from a general covariance type. Argument matches kind of apparent nested matrix, but not upper/lower.
#ifdef __cpp_concepts
    template<covariance Arg> requires
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) == square_root_covariance<Derived>) and
      (not internal::same_triangle_type_as<nested_matrix_t<Arg>, ApparentNestedMatrix>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) == square_root_covariance<Derived>) and
      (not internal::same_triangle_type_as<nested_matrix_t<Arg>, ApparentNestedMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(internal::convert_nested_matrix<NestedMatrix>(arg)),
        synchronized(true),
        apparent_nested_matrix(adjoint(std::forward<Arg>(arg).nested_matrix())) {}


    /// Construct from a general covariance type. Argument does not match apparent nested matrix.
#ifdef __cpp_concepts
    template<covariance Arg> requires
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) != square_root_covariance<Derived>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) != square_root_covariance<Derived>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(internal::convert_nested_matrix<NestedMatrix>(std::forward<Arg>(arg))),
        synchronized(false) {}


    /// Construct from a covariance nested matrix.
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)),
        synchronized(false) {}

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        synchronized = other.synchronized;
        if (synchronized) apparent_nested_matrix = other.apparent_nested_matrix;
        Base::operator=(other).nested_matrix();
      }
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        synchronized = other.synchronized;
        if (synchronized) apparent_nested_matrix = std::move(other.apparent_nested_matrix);
        Base::operator=(std::move(other).nested_matrix());
      }
      return *this;
    }

    /// Assign from a covariance_nestable.
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        Base::operator=(std::forward<Arg>(arg));
        synchronized = false;
      }
      return *this;
    }

    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr(element_settable<Derived, 2>)
        return ElementSetter(
          apparent_nested_matrix,
          i, j,
          [this] { if (not synchronized) synchronize(); },
          [this]
          {
            if constexpr(square_root_covariance<Derived>)
              nested_matrix() = Cholesky_square(apparent_nested_matrix);
            else
              nested_matrix() = Cholesky_factor(apparent_nested_matrix);
          });
      else
        return make_ElementSetter<true>(apparent_nested_matrix, i, j, [] {}, [this] { if (not synchronized) synchronize(); });
    }

    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(apparent_nested_matrix, i, j, [] {}, [this] { if (not synchronized) synchronize(); });
    }

    decltype(auto) operator[](std::size_t i) const = delete;

    decltype(auto) operator()(std::size_t i) const = delete;


  protected:
#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct CovarianceBase;


    template<typename T, typename Arg>
#ifdef __cpp_concepts
    requires (std::is_void_v<T> or covariance_nestable<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
    friend constexpr decltype(auto) convert_nested_matrix(Arg&&) noexcept;


    void mark_changed()
    {
      synchronized = false;
    }


    /// Get the apparent nested matrix.
    constexpr auto& get_apparent_nested_matrix() &
    {
      if (not synchronized) synchronize();
      return apparent_nested_matrix;
    }


    /// Get the apparent nested matrix.
    constexpr auto&& get_apparent_nested_matrix() &&
    {
      if (not synchronized) synchronize();
      return std::move(apparent_nested_matrix);
    }


    /// Get the apparent nested matrix.
    constexpr const auto& get_apparent_nested_matrix() const &
    {
      if (not synchronized) synchronize();
      return apparent_nested_matrix;
    }


    /// Get the apparent nested matrix.
    constexpr const auto&& get_apparent_nested_matrix() const &&
    {
      if (not synchronized) synchronize();
      return std::move(apparent_nested_matrix);
    }

  };


  // ============================================================================
  /**
   * \internal
   * Base of Covariance and SquareRootCovariance classes, if ArgType is an lvalue reference, and
   * # Derived is a square root and the nested matrix is not triangular (i.e., it is self-adjoint but not diagonal); or
   * # Derived is not a square root and the nested matrix is not self-adjoint (i.e., it is triangular but not diagonal).
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    (square_root_covariance<Derived> or not self_adjoint_matrix<ArgType>) and
    (not square_root_covariance<Derived> or not triangular_matrix<ArgType>) and
    (std::is_lvalue_reference_v<ArgType> or internal::contains_nested_lvalue_reference<ArgType>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    (square_root_covariance<Derived> or not self_adjoint_matrix<ArgType>) and
    (not square_root_covariance<Derived> or not triangular_matrix<ArgType>) and
    (std::is_lvalue_reference_v<ArgType> or internal::contains_nested_lvalue_reference<ArgType>)>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using NestedMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, NestedMatrix>;
    using Base::nested_matrix;
    using Scalar = typename MatrixTraits<NestedMatrix>::Scalar;

  private:
    using ApparentNestedMatrix = std::conditional_t<triangular_matrix<NestedMatrix>,
      typename MatrixTraits<NestedMatrix>::template SelfAdjointBaseType<>,
      typename MatrixTraits<NestedMatrix>::template TriangularBaseType<>>;

    const bool apparent_nested_linked;

    bool * const synchronized;

    ApparentNestedMatrix * const apparent_nested_matrix; ///< Pointer to the apparent nested matrix in another covariance.


    void synchronize() const
    {
      if constexpr(square_root_covariance<Derived>)
        *apparent_nested_matrix = Cholesky_factor(nested_matrix());
      else
        *apparent_nested_matrix = Cholesky_square(nested_matrix());
      *synchronized = true;
    }


  public:
    /// Default constructor.
    CovarianceBase() = delete;


    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base(other.nested_matrix()), apparent_nested_linked(other.apparent_nested_linked),
        synchronized(other.synchronized), apparent_nested_matrix(other.apparent_nested_matrix) {}


    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base(std::move(other.nested_matrix())), apparent_nested_linked(other.apparent_nested_linked),
        synchronized(other.synchronized), apparent_nested_matrix(std::move(other.apparent_nested_matrix)) {}


    /// Construct from another covariance that does not store a distinct apparent nested matrix.
#ifdef __cpp_concepts
    template<typename Arg> requires covariance_nestable<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>)))
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>))), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)),
        apparent_nested_linked(false),
        synchronized(new bool {false}),
        apparent_nested_matrix(new ApparentNestedMatrix) {}


    /// Construct from a covariance_nestable or another covariance that stores a distinct apparent nested matrix (nested matrix is not an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>))) and
      (not (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>))
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
        ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
          (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>)) and not
        (internal::contains_nested_lvalue_reference<Arg> or
          internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix())),
        apparent_nested_linked(true), synchronized(&arg.synchronized),
        apparent_nested_matrix(&arg.apparent_nested_matrix) {}


    /// Construct from another covariance that stores a distinct apparent nested matrix (nested matrix is an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>))) and
      (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
      ((self_adjoint_matrix<nested_matrix_t<Arg>> and not square_root_covariance<Arg>) or
        (triangular_matrix<nested_matrix_t<Arg>> and square_root_covariance<Arg>)) and
      (internal::contains_nested_lvalue_reference<Arg> or
        internal::contains_nested_lvalue_reference<nested_matrix_t<Arg>>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<decltype(arg.nested_matrix())>(arg.nested_matrix())),
        apparent_nested_linked(true), synchronized(arg.synchronized),
        apparent_nested_matrix(std::forward<Arg>(arg).apparent_nested_matrix) {}


    ~CovarianceBase()
    {
      if (not apparent_nested_linked)
      {
        delete apparent_nested_matrix;
        delete synchronized;
      }
    }


    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        *synchronized = *other.synchronized;
        if (*synchronized) *apparent_nested_matrix = *other.apparent_nested_matrix;
        Base::operator=(other).nested_matrix();
      }
      return *this;
    }


    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<NestedMatrix> and not identity_matrix<NestedMatrix>) if (this != &other)
      {
        *synchronized = *other.synchronized;
        if (*synchronized) *apparent_nested_matrix = std::move(*other.apparent_nested_matrix);
        Base::operator=(std::move(other).nested_matrix());
      }
      return *this;
    }


    /// Assign from a covariance_nestable.
#ifdef __cpp_concepts
    template<covariance_nestable Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_nestable<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<NestedMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<NestedMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        Base::operator=(std::forward<Arg>(arg));
        *synchronized = false;
      }
      return *this;
    }


    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr(element_settable<Derived, 2>)
        return ElementSetter(
          *apparent_nested_matrix,
          i, j,
          [this] { if (not *synchronized) synchronize(); },
          [this]
          {
            if constexpr(square_root_covariance<Derived>)
              nested_matrix() = Cholesky_square(*apparent_nested_matrix);
            else
              nested_matrix() = Cholesky_factor(*apparent_nested_matrix);
          });
      else
        return make_ElementSetter<true>(*apparent_nested_matrix, i, j, [] {}, [this] { if (not *synchronized) synchronize(); });
    }


    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(*apparent_nested_matrix, i, j, [] {}, [this] { if (not *synchronized) synchronize(); });
    }


    decltype(auto) operator[](std::size_t i) const = delete;


    decltype(auto) operator()(std::size_t i) const = delete;


  protected:
#ifdef __cpp_concepts
    template<typename, typename>
#else
    template<typename, typename, typename>
#endif
    friend struct CovarianceBase;


    template<typename T, typename Arg>
#ifdef __cpp_concepts
    requires (std::is_void_v<T> or covariance_nestable<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
    friend constexpr decltype(auto) convert_nested_matrix(Arg&&) noexcept;


    void mark_changed()
    {
      *synchronized = false;
    }


    /// Get the apparent nested matrix.
    constexpr auto& get_apparent_nested_matrix() &
    {
      if (not *synchronized) synchronize();
      return *apparent_nested_matrix;
    }


    /// Get the apparent nested matrix.
    constexpr auto&& get_apparent_nested_matrix() &&
    {
      if (not *synchronized) synchronize();
      return std::move(*apparent_nested_matrix);
    }


    /// Get the apparent nested matrix.
    constexpr const auto& get_apparent_nested_matrix() const &
    {
      if (not *synchronized) synchronize();
      return *apparent_nested_matrix;
    }


    /// Get the apparent nested matrix.
    constexpr const auto&& get_apparent_nested_matrix() const &&
    {
      if (not *synchronized) synchronize();
      return std::move(*apparent_nested_matrix);
    }

  };

}

#endif //OPENKALMAN_COVARIANCEBASE_H
