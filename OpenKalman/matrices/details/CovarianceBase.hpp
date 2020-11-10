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
  // ============================================================================
  /*
   * Base of Covariance and SquareRootCovariance classes, if ArgType is not an lvalue reference and either
   * (1) Derived is not a square root and the base is self-adjoint; or
   * (2) Derived is a square root and the base is triangular.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    ((self_adjoint_matrix<ArgType> and not square_root_covariance<Derived>) or
      (triangular_matrix<ArgType> and square_root_covariance<Derived>)) and
    (not (std::is_lvalue_reference_v<ArgType> or std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>))
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    ((self_adjoint_matrix<ArgType> and not square_root_covariance<Derived>) or
    (triangular_matrix<ArgType> and square_root_covariance<Derived>)) and not
    (std::is_lvalue_reference_v<ArgType> or
    std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>)>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using BaseMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, BaseMatrix>;
    using Base::Base;
    using Base::operator=;
    using Base::base_matrix;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;

    auto operator() (std::size_t i, std::size_t j)
    {
      return make_ElementSetter<not is_element_settable_v<Derived, 2>>(base_matrix(), i, j);
    }

    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(base_matrix(), i, j);
    }

    auto operator[] (std::size_t i)
    {
      return make_ElementSetter<not is_element_settable_v<Derived, 2>>(base_matrix(), i);
    }

    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(base_matrix(), i);
    }

    auto operator() (std::size_t i) { return operator[](i); }

    auto operator() (std::size_t i) const { return operator[](i); }

  protected:
    template<typename, typename Arg>
    friend constexpr decltype(auto) convert_base_matrix(Arg&&) noexcept;

    constexpr void mark_changed() const {}

    /// Get the apparent base matrix.
    constexpr auto& get_apparent_base_matrix() & { return base_matrix(); }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() && { return std::move(base_matrix()); }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const & { return base_matrix(); }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const && { return std::move(base_matrix()); }

  };


  // ============================================================================
  /*
   * Base of Covariance and SquareRootCovariance classes, if ArgType is an lvalue reference.
   * No conversion is necessary if either
   * (1) Derived is not a square root and the base is self-adjoint; or
   * (2) Derived is a square root and the base is triangular.
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    ((self_adjoint_matrix<ArgType> and not square_root_covariance<Derived>) or
      (triangular_matrix<ArgType> and square_root_covariance<Derived>)) and
    (std::is_lvalue_reference_v<ArgType> or std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    ((self_adjoint_matrix<ArgType> and not square_root_covariance<Derived>) or
    (triangular_matrix<ArgType> and square_root_covariance<Derived>)) and
    (std::is_lvalue_reference_v<ArgType> or
    std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>)>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using BaseMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, BaseMatrix>;
    using Base::base_matrix;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;

  private:
    const bool apparent_base_linked;

    bool * const synchronized;

  public:
    /// Default constructor.
    CovarianceBase() = delete;

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base(other.base_matrix()), synchronized(other.synchronized) {}

    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base(std::move(other)), synchronized(other.synchronized) {}

    /// Construct from a covariance base or another covariance that does not store a distinct apparent base matrix.
#ifdef __cpp_concepts
    template<typename Arg> requires covariance_base<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)))
#else
    template<typename Arg,
      std::enable_if_t<covariance_base<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>))), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)), apparent_base_linked(false), synchronized(new bool {true}) {}

    /// Construct from another covariance that stores a distinct apparent base matrix (base matrix is not an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>))) and
      (not (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
        std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>))
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
      ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
      (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and not
      (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
      std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)), apparent_base_linked(true), synchronized(&arg.synchronized) {}

    /// Construct from another covariance that stores a distinct apparent base matrix (base matrix is an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>))) and
      (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
        std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
      ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
      (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and
      (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
      std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)), apparent_base_linked(true), synchronized(arg.synchronized) {}

    ~CovarianceBase()
    {
      if (not apparent_base_linked)
      {
        delete synchronized;
      }
    }

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
      {
        Base::operator=(other);
        if (apparent_base_linked) *synchronized = false;
      }
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
      {
        Base::operator=(std::move(other));
        if (apparent_base_linked) *synchronized = false;
      }
      return *this;
    }

    /// Assign from a covariance base or typed matrix base.
#ifdef __cpp_concepts
    template<typename Arg> requires (covariance_base<Arg> or typed_matrix_base<Arg>) and
      std::is_assignable_v<BaseMatrix, Arg&&>
#else
    template<typename Arg, std::enable_if_t<(covariance_base<Arg> or typed_matrix_base<Arg>) and
      std::is_assignable_v<BaseMatrix, Arg&&>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
      {
        static_assert(identity_matrix<Arg>);
      }
      else
      {
        Base::operator=(std::forward<Arg>(arg));
        if (apparent_base_linked) *synchronized = false;
      }
      return *this;
    }

    auto operator() (std::size_t i, std::size_t j)
    {
      if constexpr (is_element_settable_v<Derived, 2>)
        return ElementSetter(base_matrix(), i, j, [] {}, [this] { if (apparent_base_linked) *synchronized = false; });
      else
        return make_ElementSetter<true>(base_matrix(), i, j);
    }

    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(base_matrix(), i, j);
    }

    auto operator[] (std::size_t i)
    {
      if constexpr (is_element_settable_v<Derived, 1>)
        return ElementSetter(base_matrix(), i, [] {}, [this] { if (apparent_base_linked) *synchronized = false; });
      else
        return make_ElementSetter<true>(base_matrix(), i);
    }

    auto operator[] (std::size_t i) const
    {
      return make_ElementSetter<true>(base_matrix(), i);
    }

    auto operator() (std::size_t i) { return operator[](i); }

    auto operator() (std::size_t i) const { return operator[](i); }

  protected:
    template<typename, typename Arg>
    friend constexpr decltype(auto) convert_base_matrix(Arg&&) noexcept;

    constexpr void mark_changed() const {}

    /// Get the apparent base matrix.
    constexpr auto& get_apparent_base_matrix() & { return base_matrix(); }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() && { return std::move(base_matrix()); }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const & { return base_matrix(); }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const && { return std::move(base_matrix()); }

  };


  // ============================================================================
  /**
   * Base of Covariance and SquareRootCovariance classes, if ArgType is not an lvalue reference, and
   * (1) Derived is a square root and the base is not triangular (i.e., it is self-adjoint but not diagonal); or
   * (2) Derived is not a square root and the base is not self-adjoint (i.e., it is triangular but not diagonal).
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    (not self_adjoint_matrix<ArgType> or square_root_covariance<Derived>) and
    (not triangular_matrix<ArgType> or not square_root_covariance<Derived>) and
    (not std::is_lvalue_reference_v<ArgType>) and
    (not std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    (not self_adjoint_matrix<ArgType> or square_root_covariance<Derived>) and
    (not triangular_matrix<ArgType> or not square_root_covariance<Derived>) and
    not std::is_lvalue_reference_v<ArgType> and
    not std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using BaseMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, BaseMatrix>;
    using Base::base_matrix;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;

  private:
    using ApparentBaseMatrix = std::conditional_t<triangular_matrix<BaseMatrix>,
      typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<>,
      typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>;

    mutable bool synchronized;

    mutable ApparentBaseMatrix apparent_base; ///< The apparent base matrix for Covariance or SquareRootCovariance.

    void synchronize() const
    {
      if constexpr(square_root_covariance<Derived>)
        apparent_base = Cholesky_factor(base_matrix());
      else
        apparent_base = Cholesky_square(base_matrix());
      synchronized = true;
    }

  public:
    /// Default constructor.
    CovarianceBase() : synchronized() {}

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base(other),
        synchronized(other.synchronized),
        apparent_base(other.apparent_base) {}

    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base(std::move(other.base_matrix())),
        synchronized(other.synchronized),
        apparent_base(std::move(other.apparent_base)) {}

    /// Construct from a general covariance type. Argument matches apparent base.
#ifdef __cpp_concepts
    template<covariance Arg> requires
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) == square_root_covariance<Derived>) and
      (upper_triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> == upper_triangular_matrix<ApparentBaseMatrix>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) == square_root_covariance<Derived> and
      (upper_triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> == upper_triangular_matrix<ApparentBaseMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(internal::convert_base_matrix<BaseMatrix>(arg)),
        synchronized(true),
        apparent_base(std::forward<Arg>(arg).base_matrix()) {}

    /// Construct from a general covariance type. Argument matches kind of apparent base, but not upper/lower.
#ifdef __cpp_concepts
    template<covariance Arg> requires
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) == square_root_covariance<Derived>) and
      (upper_triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> != upper_triangular_matrix<ApparentBaseMatrix>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) == square_root_covariance<Derived> and
      (upper_triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> != upper_triangular_matrix<ApparentBaseMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(internal::convert_base_matrix<BaseMatrix>(arg)),
        synchronized(true),
        apparent_base(adjoint(std::forward<Arg>(arg).base_matrix())) {}

    /// Construct from a general covariance type. Argument does not match apparent base.
#ifdef __cpp_concepts
    template<covariance Arg> requires
      ((cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) != square_root_covariance<Derived>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and
      (cholesky_form<Arg> or (diagonal_matrix<Arg> and square_root_covariance<Arg>)) != square_root_covariance<Derived>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(internal::convert_base_matrix<BaseMatrix>(std::forward<Arg>(arg))),
        synchronized(false) {}

    /// Construct from a covariance base matrix.
#ifdef __cpp_concepts
    template<covariance_base Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_base<Arg>, int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)),
        synchronized(false) {}

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
      {
        synchronized = other.synchronized;
        if (synchronized) apparent_base = other.apparent_base;
        Base::operator=(other).base_matrix();
      }
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
      {
        synchronized = other.synchronized;
        if (synchronized) apparent_base = std::move(other.apparent_base);
        Base::operator=(std::move(other).base_matrix());
      }
      return *this;
    }

    /// Assign from a covariance base.
#ifdef __cpp_concepts
    template<covariance_base Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_base<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg) noexcept
    {
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
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
      if constexpr(is_element_settable_v<Derived, 2>)
        return ElementSetter(
          apparent_base,
          i, j,
          [this] { if (not synchronized) synchronize(); },
          [this]
          {
            if constexpr(square_root_covariance<Derived>)
              base_matrix() = Cholesky_square(apparent_base);
            else
              base_matrix() = Cholesky_factor(apparent_base);
          });
      else
        return make_ElementSetter<true>(apparent_base, i, j, [] {}, [this] { if (not synchronized) synchronize(); });
    }

    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(apparent_base, i, j, [] {}, [this] { if (not synchronized) synchronize(); });
    }

    decltype(auto) operator[](std::size_t i) const = delete;

    decltype(auto) operator()(std::size_t i) const = delete;


  protected:
    template<typename, typename, typename>
    friend struct CovarianceBase;

    template<typename, typename Arg>
    friend constexpr decltype(auto) convert_base_matrix(Arg&&) noexcept;

    void mark_changed()
    {
      synchronized = false;
    }

    /// Get the apparent base matrix.
    constexpr auto& get_apparent_base_matrix() &
    {
      if (not synchronized) synchronize();
      return apparent_base;
    }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() &&
    {
      if (not synchronized) synchronize();
      return std::move(apparent_base);
    }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const &
    {
      if (not synchronized) synchronize();
      return apparent_base;
    }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const &&
    {
      if (not synchronized) synchronize();
      return std::move(apparent_base);
    }

  };


  // ============================================================================
  /**
   * Base of Covariance and SquareRootCovariance classes, if ArgType is an lvalue reference, and
   * (1) Derived is a square root and the base is not triangular (i.e., it is self-adjoint but not diagonal); or
   * (2) Derived is not a square root and the base is not self-adjoint (i.e., it is triangular but not diagonal).
   */
#ifdef __cpp_concepts
  template<typename Derived, typename ArgType> requires
    (not self_adjoint_matrix<ArgType> or square_root_covariance<Derived>) and
    (not triangular_matrix<ArgType> or not square_root_covariance<Derived>) and
    (std::is_lvalue_reference_v<ArgType> or std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>)
  struct CovarianceBase<Derived, ArgType>
#else
  template<typename Derived, typename ArgType>
  struct CovarianceBase<Derived, ArgType, std::enable_if_t<
    (not self_adjoint_matrix<ArgType> or square_root_covariance<Derived>) and
    (not triangular_matrix<ArgType> or not square_root_covariance<Derived>) and
    (std::is_lvalue_reference_v<ArgType> or
    std::is_lvalue_reference_v<typename MatrixTraits<ArgType>::BaseMatrix>)>>
#endif
  : CovarianceBaseBase<Derived, ArgType>
  {
    using BaseMatrix = ArgType;
    using Base = CovarianceBaseBase<Derived, BaseMatrix>;
    using Base::base_matrix;
    using Scalar = typename MatrixTraits<BaseMatrix>::Scalar;

  private:
    using ApparentBaseMatrix = std::conditional_t<triangular_matrix<BaseMatrix>,
      typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<>,
      typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>;

    const bool apparent_base_linked;

    bool * const synchronized;

    ApparentBaseMatrix * const apparent_base; ///< Pointer to the apparent base matrix in another covariance.

    void synchronize() const
    {
      if constexpr(square_root_covariance<Derived>)
        *apparent_base = Cholesky_factor(base_matrix());
      else
        *apparent_base = Cholesky_square(base_matrix());
      *synchronized = true;
    }

  public:
    /// Default constructor.
    CovarianceBase() = delete;

    /// Copy constructor.
    CovarianceBase(const CovarianceBase& other)
      : Base(other.base_matrix()), apparent_base_linked(other.apparent_base_linked),
        synchronized(other.synchronized), apparent_base(other.apparent_base) {}

    /// Move constructor.
    CovarianceBase(CovarianceBase&& other) noexcept
      : Base(std::move(other.base_matrix())), apparent_base_linked(other.apparent_base_linked),
        synchronized(other.synchronized), apparent_base(std::move(other.apparent_base)) {}

    /// Construct from another covariance that does not store a distinct apparent base matrix.
#ifdef __cpp_concepts
    template<typename Arg> requires covariance_base<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)))
#else
    template<typename Arg, std::enable_if_t<covariance_base<Arg> or (covariance<Arg> and
        ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>))), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<Arg>(arg)),
        apparent_base_linked(false),
        synchronized(new bool {false}),
        apparent_base(new ApparentBaseMatrix) {}

    /// Construct from a covariance base or another covariance that stores a distinct apparent base matrix (base matrix is not an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>))) and
      (not (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
        std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>))
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
        ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and not
        (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
        std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<decltype(arg.base_matrix())>(arg.base_matrix())),
        apparent_base_linked(true), synchronized(&arg.synchronized),
        apparent_base(&arg.apparent_base) {}

    /// Construct from another covariance that stores a distinct apparent base matrix (base matrix is an lvalue ref).
#ifdef __cpp_concepts
    template<covariance Arg> requires
      (not ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
        (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>))) and
      (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
        std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>)
#else
    template<typename Arg, std::enable_if_t<covariance<Arg> and not
      ((self_adjoint_matrix<typename MatrixTraits<Arg>::BaseMatrix> and not square_root_covariance<Arg>) or
      (triangular_matrix<typename MatrixTraits<Arg>::BaseMatrix> and square_root_covariance<Arg>)) and
      (std::is_lvalue_reference_v<typename MatrixTraits<Arg>::BaseMatrix> or
      std::is_lvalue_reference_v<typename MatrixTraits<typename MatrixTraits<Arg>::BaseMatrix>::BaseMatrix>), int> = 0>
#endif
    CovarianceBase(Arg&& arg) noexcept
      : Base(std::forward<decltype(arg.base_matrix())>(arg.base_matrix())),
        apparent_base_linked(true), synchronized(arg.synchronized),
        apparent_base(std::forward<Arg>(arg).apparent_base) {}

    ~CovarianceBase()
    {
      if (not apparent_base_linked)
      {
        delete apparent_base;
        delete synchronized;
      }
    }

    /// Copy assignment operator.
    auto& operator=(const CovarianceBase& other)
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
      {
        *synchronized = *other.synchronized;
        if (*synchronized) *apparent_base = *other.apparent_base;
        Base::operator=(other).base_matrix();
      }
      return *this;
    }

    /// Move assignment operator.
    auto& operator=(CovarianceBase&& other) noexcept
    {
      if constexpr (not zero_matrix<BaseMatrix> and not identity_matrix<BaseMatrix>) if (this != &other)
      {
        *synchronized = *other.synchronized;
        if (*synchronized) *apparent_base = std::move(*other.apparent_base);
        Base::operator=(std::move(other).base_matrix());
      }
      return *this;
    }

    /// Assign from a covariance base.
#ifdef __cpp_concepts
    template<covariance_base Arg>
#else
    template<typename Arg, std::enable_if_t<covariance_base<Arg>, int> = 0>
#endif
    auto& operator=(Arg&& arg)
    {
      if constexpr (zero_matrix<BaseMatrix>)
      {
        static_assert(zero_matrix<Arg>);
      }
      else if constexpr (identity_matrix<BaseMatrix>)
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
      if constexpr(is_element_settable_v<Derived, 2>)
        return ElementSetter(
          *apparent_base,
          i, j,
          [this] { if (not *synchronized) synchronize(); },
          [this]
          {
            if constexpr(square_root_covariance<Derived>)
              base_matrix() = Cholesky_square(*apparent_base);
            else
              base_matrix() = Cholesky_factor(*apparent_base);
          });
      else
        return make_ElementSetter<true>(*apparent_base, i, j, [] {}, [this] { if (not *synchronized) synchronize(); });
    }

    auto operator() (std::size_t i, std::size_t j) const
    {
      return make_ElementSetter<true>(*apparent_base, i, j, [] {}, [this] { if (not *synchronized) synchronize(); });
    }

    decltype(auto) operator[](std::size_t i) const = delete;

    decltype(auto) operator()(std::size_t i) const = delete;


  protected:
    template<typename, typename, typename>
    friend struct CovarianceBase;

    template<typename, typename Arg>
    friend constexpr decltype(auto) convert_base_matrix(Arg&&) noexcept;

    void mark_changed()
    {
      *synchronized = false;
    }

    /// Get the apparent base matrix.
    constexpr auto& get_apparent_base_matrix() &
    {
      if (not *synchronized) synchronize();
      return *apparent_base;
    }

    /// Get the apparent base matrix.
    constexpr auto&& get_apparent_base_matrix() &&
    {
      if (not *synchronized) synchronize();
      return std::move(*apparent_base);
    }

    /// Get the apparent base matrix.
    constexpr const auto& get_apparent_base_matrix() const &
    {
      if (not *synchronized) synchronize();
      return *apparent_base;
    }

    /// Get the apparent base matrix.
    constexpr const auto&& get_apparent_base_matrix() const &&
    {
      if (not *synchronized) synchronize();
      return std::move(*apparent_base);
    }

  };

}

#endif //OPENKALMAN_COVARIANCEBASE_H
